#include "sparkium/pipelines/portable/transpile.h"

#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <set>
#include <sstream>
#include <stdexcept>

namespace sparkium::portable {
namespace {

bool IsIdent(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// ---------------------------------------------------------------------------
// Lightweight C tokenizer shared by the include expander and the rewriter.
// ---------------------------------------------------------------------------
enum TokenKind { TK_IDENT, TK_NUMBER, TK_STRING, TK_CHAR, TK_PUNCT, TK_COMMENT, TK_DIRECTIVE, TK_WS };

struct Token {
  TokenKind kind;
  std::string text;
};

std::vector<Token> Tokenize(const std::string &text) {
  std::vector<Token> tokens;
  size_t i = 0;
  bool line_start = true;
  while (i < text.size()) {
    const char c = text[i];
    if (c == '\n') {
      tokens.push_back({TK_WS, "\n"});
      ++i;
      line_start = true;
      continue;
    }
    if (c == ' ' || c == '\t' || c == '\r') {
      size_t j = i;
      while (j < text.size() && (text[j] == ' ' || text[j] == '\t' || text[j] == '\r'))
        ++j;
      tokens.push_back({TK_WS, text.substr(i, j - i)});
      i = j;
      continue;
    }
    if (c == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      size_t j = text.find('\n', i);
      if (j == std::string::npos)
        j = text.size();
      tokens.push_back({TK_COMMENT, text.substr(i, j - i)});
      i = j;
      continue;
    }
    if (c == '/' && i + 1 < text.size() && text[i + 1] == '*') {
      size_t j = text.find("*/", i + 2);
      if (j == std::string::npos)
        j = text.size() - 2;
      tokens.push_back({TK_COMMENT, text.substr(i, j + 2 - i)});
      i = j + 2;
      continue;
    }
    if (c == '#' && line_start) {
      size_t j = text.find('\n', i);
      if (j == std::string::npos)
        j = text.size();
      // Handle line continuations.
      while (j > 0 && j < text.size() && text[j - 1] == '\\') {
        j = text.find('\n', j + 1);
        if (j == std::string::npos)
          j = text.size();
      }
      tokens.push_back({TK_DIRECTIVE, text.substr(i, j - i)});
      i = j;
      continue;
    }
    line_start = false;
    if (c == '"' || c == '\'') {
      size_t j = i + 1;
      while (j < text.size() && text[j] != c) {
        if (text[j] == '\\')
          ++j;
        ++j;
      }
      tokens.push_back({c == '"' ? TK_STRING : TK_CHAR, text.substr(i, j + 1 - i)});
      i = j + 1;
      continue;
    }
    if (IsIdent(c) && !std::isdigit(static_cast<unsigned char>(c))) {
      size_t j = i;
      while (j < text.size() && IsIdent(text[j]))
        ++j;
      tokens.push_back({TK_IDENT, text.substr(i, j - i)});
      i = j;
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && i + 1 < text.size() &&
                                                        std::isdigit(static_cast<unsigned char>(text[i + 1])))) {
      size_t j = i;
      while (j < text.size() &&
             (std::isalnum(static_cast<unsigned char>(text[j])) || text[j] == '.' || text[j] == '_' ||
              ((text[j] == '+' || text[j] == '-') && j > i && (text[j - 1] == 'e' || text[j - 1] == 'E'))))
        ++j;
      tokens.push_back({TK_NUMBER, text.substr(i, j - i)});
      i = j;
      continue;
    }
    // Multi-character punctuation.
    static const char *kMulti[] = {"<<=", ">>=", "...", "++", "--", "+=", "-=", "*=", "/=", "%=",
                                   "==", "!=", "<=", ">=", "&&", "||", "<<", ">>", "&=", "|=", "^=",
                                   "->", "::", ".*"};
    size_t matched = 0;
    for (const char *m : kMulti) {
      const size_t len = std::strlen(m);
      if (text.compare(i, len, m) == 0) {
        matched = len;
        break;
      }
    }
    if (matched == 0)
      matched = 1;
    tokens.push_back({TK_PUNCT, text.substr(i, matched)});
    i += matched;
  }
  return tokens;
}

// True if the token sequence starting at `dot` (the '.' token) is a swizzle
// access (.xy, .r, .bgr, ...). Only xyzw/rgba members of length <= 4 where the
// preceding token can end an lvalue expression.
bool IsSwizzle(const std::vector<Token> &tokens, size_t dot) {
  // Swizzles never appear in qualified names (ns::field) or preprocessor
  // output; a preceding ':' means this is not a member access on a vector.
  for (size_t k = dot; k-- > 0;) {
    if (tokens[k].kind == TK_WS)
      continue;
    if (tokens[k].kind == TK_PUNCT && tokens[k].text == ":")
      return false;
    break;
  }
  if (dot + 1 >= tokens.size() || tokens[dot + 1].kind != TK_IDENT)
    return false;
  const std::string &name = tokens[dot + 1].text;
  if (name.empty() || name.size() > 4)
    return false;
  bool has_xyzw = false, has_rgba = false;
  for (char c : name) {
    if (std::string("xyzw").find(c) != std::string::npos)
      has_xyzw = true;
    else if (std::string("rgba").find(c) != std::string::npos)
      has_rgba = true;
    else
      return false;
  }
  if (has_xyzw && has_rgba)
    return false;
  if (dot == 0)
    return false;
  const Token &prev = tokens[dot - 1];
  if (prev.kind == TK_IDENT || prev.kind == TK_NUMBER)
    return true;
  if (prev.kind == TK_PUNCT && (prev.text == ")" || prev.text == "]"))
    return true;
  return false;
}

// Find the start of the primary expression that ends at token index `end`
// (exclusive), walking backwards over balanced (), [] and postfix .member /
// function-call chains.
size_t ExpressionStart(const std::vector<Token> &tokens, size_t end) {
  size_t i = end;
  while (i > 0) {
    const Token &t = tokens[i - 1];
    if (t.kind == TK_PUNCT && (t.text == ")" || t.text == "]")) {
      const std::string open = t.text == ")" ? "(" : "[";
      const std::string close = t.text;
      int depth = 1;
      --i;
      while (i > 0 && depth > 0) {
        --i;
        if (tokens[i].kind == TK_PUNCT) {
          if (tokens[i].text == close)
            ++depth;
          else if (tokens[i].text == open)
            --depth;
        }
      }
      continue;
    }
    if (t.kind == TK_IDENT || t.kind == TK_NUMBER) {
      --i;
      continue;
    }
    if (t.kind == TK_PUNCT && t.text == ".") {
      --i;
      continue;
    }
    break;
  }
  return i;
}

std::string TrimCopy(std::string s) {
  const auto b = s.find_first_not_of(" \t");
  if (b == std::string::npos)
    return std::string();
  const auto e = s.find_last_not_of(" \t");
  return s.substr(b, e - b + 1);
}

std::string JoinTokens(const std::vector<Token> &tokens, size_t begin, size_t end) {
  std::string out;
  for (size_t i = begin; i < end; ++i)
    out += tokens[i].text;
  return out;
}

// The multi-component swizzle expansion turns HLSL asfloat(v.xyz) into
// asfloat(float3(v[0], v[1], v[2])). In HLSL the outer asfloat bit-casts each
// component, but the generated C++ would construct float3 from the integer
// components and convert them numerically. Rewrite
// asfloat(floatN(a, b, ...)) into SparkiumAsFloat(a, b, ...), which bit-casts
// every component (see hlsl_compat.h).
std::string RewriteAsFloatVectorCasts(const std::string &text) {
  const std::string prefix = "asfloat(float";
  std::string out;
  out.reserve(text.size());
  size_t i = 0;
  while (i < text.size()) {
    const size_t found = text.find(prefix, i);
    if (found == std::string::npos) {
      out += text.substr(i);
      break;
    }
    out += text.substr(i, found - i);
    size_t p = found + prefix.size();
    // Expect a single digit then '('.
    if (p >= text.size() || !std::isdigit(static_cast<unsigned char>(text[p]))) {
      out += text.substr(found, prefix.size());
      i = found + prefix.size();
      continue;
    }
    ++p;
    if (p >= text.size() || text[p] != '(') {
      out += text.substr(found, prefix.size());
      i = found + prefix.size();
      continue;
    }
    ++p;
    const size_t args_begin = p;
    // Scan for the ')' that closes the floatN(...) argument list. depth starts
    // at 1 to account for the floatN '(' we just consumed; inner parens nest.
    int depth = 1;
    while (p < text.size() && depth > 0) {
      if (text[p] == '(')
        ++depth;
      else if (text[p] == ')')
        --depth;
      ++p;
    }
    if (depth != 0) {
      out += text.substr(found, prefix.size());
      i = found + prefix.size();
      continue;
    }
    // p is now just past the floatN ')'. Require the asfloat ')' immediately
    // after; otherwise this is not the asfloat(floatN(...)) idiom.
    if (p >= text.size() || text[p] != ')') {
      out += text.substr(found, prefix.size());
      i = found + prefix.size();
      continue;
    }
    const std::string args = text.substr(args_begin, p - 1 - args_begin);
    out += "SparkiumAsFloat(" + args + ")";
    i = p + 1;  // also consume the asfloat ')'
  }
  return out;
}

// Detect function definitions and prefix them with SPARKIUM_KERNEL so they
// compile as CUDA device functions. Works on the joined output string where
// every statement is terminated by ';' and every block by '}' at brace depth
// 0, so a candidate header runs from the last such terminator to the '{' that
// opens a body. Anything in between that is not a valid function declarator
// (variable definitions, control statements, class bodies, ...) is rejected.
std::string AnnotateDeviceFunctions(const std::string &text) {
  std::string out;
  out.reserve(text.size() + text.size() / 8);
  int depth = 0;
  size_t header_start = 0;  // start of the current top-level "header" run
  size_t cursor = 0;        // how far we have copied into out
  size_t i = 0;
  while (i < text.size()) {
    const char c = text[i];
    if (c == '{') {
      bool function_def = false;
      size_t insert_pos = 0;
      if (true) {
        // Header text is everything since the last top-level ';' or '}'.
        std::string raw_header = text.substr(header_start, i - header_start);
        // Strip comments so commented-out code between declarations does not
        // break the header shape.
        std::string header;
        header.reserve(raw_header.size());
        // Byte offset (in raw_header) of the last comment end; annotations
        // are never inserted inside a comment.
        size_t last_comment_end = 0;
        for (size_t q = 0; q < raw_header.size(); ++q) {
          if (raw_header[q] == '/' && q + 1 < raw_header.size() && raw_header[q + 1] == '/') {
            const size_t nl = raw_header.find('\n', q);
            if (nl == std::string::npos)
              break;
            header += '\n';
            last_comment_end = nl + 1;
            q = nl;
            continue;
          }
          if (raw_header[q] == '/' && q + 1 < raw_header.size() && raw_header[q + 1] == '*') {
            const size_t end = raw_header.find("*/", q + 2);
            q = end == std::string::npos ? raw_header.size() : end + 1;
            last_comment_end = q + 1;
            header += ' ';
            continue;
          }
          header += raw_header[q];
        }
        // Trim trailing whitespace.
        const auto hend = header.find_last_not_of(" \t\r\n");
        if (hend != std::string::npos) {
          header.erase(hend + 1);
          const auto hbegin = header.find_first_not_of(" \t\r\n");
          std::string h = hbegin == std::string::npos ? std::string() : header.substr(hbegin);
          // A function header ends with ')' (the parameter list) and must
          // not be a control statement, class/struct/namespace block, or a
          // declaration with an initializer ('=') or ','.
          if (!h.empty() && h.back() == ')') {
            // Extract the function name: identifier immediately before the
            // matching '('.
            size_t close = h.size() - 1;
            int pdepth = 1;
            size_t open = close;
            while (open > 0 && pdepth > 0) {
              --open;
              if (h[open] == ')')
                ++pdepth;
              else if (h[open] == '(')
                --pdepth;
            }
            if (pdepth == 0 && open > 0) {
              size_t name_end = open;
              while (name_end > 0 && (h[name_end - 1] == ' ' || h[name_end - 1] == '\t'))
                --name_end;
              size_t name_begin = name_end;
              while (name_begin > 0 &&
                     (std::isalnum(static_cast<unsigned char>(h[name_begin - 1])) || h[name_begin - 1] == '_' ||
                      h[name_begin - 1] == '~'))
                --name_begin;
              const std::string name = h.substr(name_begin, name_end - name_begin);
              static const std::set<std::string> kNotFunction = {
                  "if",       "while",  "for",      "switch",   "catch", "return",
                  "sizeof",   "alignof", "decltype", "case",    "do",    "else",
                  "template", "inline", "constexpr", "static",  "extern"};
              const std::string prefix = h.substr(0, name_begin);
              // `class`/`typename` may legitimately appear in a `template
              // <class T>` prefix, so only the part before `template` decides
              // whether this is a type/namespace block instead of a function.
              const auto template_pos = prefix.find("template");
              const std::string pre_template =
                  template_pos == std::string::npos ? prefix : prefix.substr(0, template_pos);
              auto has_word = [](const std::string &s, const char *word) {
                const size_t len = std::strlen(word);
                size_t pos = 0;
                while ((pos = s.find(word, pos)) != std::string::npos) {
                  const bool left_ok =
                      pos == 0 || !(std::isalnum(static_cast<unsigned char>(s[pos - 1])) || s[pos - 1] == '_');
                  const size_t after = pos + len;
                  const bool right_ok =
                      after >= s.size() ||
                      !(std::isalnum(static_cast<unsigned char>(s[after])) || s[after] == '_');
                  if (left_ok && right_ok)
                    return true;
                  pos = after;
                }
                return false;
              };
              const bool excluded = name.empty() || kNotFunction.count(name) || has_word(pre_template, "namespace") ||
                                    has_word(pre_template, "struct") || has_word(pre_template, "class") ||
                                    has_word(pre_template, "union") || has_word(pre_template, "enum");
              if (!excluded) {
                function_def = true;
                // For function templates, `__device__` must follow the
                // `template <...>` prefix (attributes cannot precede it).
                const auto htemplate_pos = h.find("template");
                if (htemplate_pos != std::string::npos && htemplate_pos < name_begin) {
                  size_t angle_end = h.find('>', htemplate_pos);
                  insert_pos = header_start + hbegin +
                               (angle_end == std::string::npos ? htemplate_pos : angle_end + 1);
                } else {
                  insert_pos = header_start + hbegin;
                }
                // Never insert inside a comment: when comments precede the
                // header, the annotation goes right after the last one.
                if (insert_pos < header_start + last_comment_end)
                  insert_pos = header_start + last_comment_end;
              }
            }
          }
        }
      }
      if (function_def) {
        // Copy up to the header start, then emit the annotation once.
        out.append(text, cursor, insert_pos - cursor);
        out += "SPARKIUM_KERNEL ";
        cursor = insert_pos;
      }
      ++depth;
      ++i;
      continue;
    }
    if (c == '}') {
      --depth;
      header_start = i + 1;
      ++i;
      continue;
    }
    if (c == ';') {
      header_start = i + 1;
      ++i;
      continue;
    }
    ++i;
  }
  out.append(text, cursor, std::string::npos);
  return out;
}

}  // namespace

Transpiler::Transpiler(const grassland::VirtualFileSystem &vfs) : vfs_(vfs) {
}

std::string Transpiler::Process(const std::string &path, const std::map<std::string, std::string> &defines) {
  // Merge (not replace) so include guards set by earlier Process() calls keep
  // deduplicating shared headers across the translation unit.
  for (const auto &[name, value] : defines)
    defines_[name] = value;
  defines_["SPARKIUM_PORTABLE"] = "1";
  return Transform(ProcessFile(path, defines_));
}

std::string Transpiler::TransformSnippet(const std::string &source) {
  std::map<std::string, std::string> defines;
  defines["SPARKIUM_PORTABLE"] = "1";
  return Transform(source);
}

std::string Transpiler::ProcessFile(const std::string &path, std::map<std::string, std::string> &defines) {
  std::vector<uint8_t> data;
  if (vfs_.ReadFile(path, data) != 0)
    throw std::runtime_error("portable transpiler: cannot read shader file: " + path);
  std::string text(data.begin(), data.end());

  struct Frame {
    bool parent_active;
    bool branch_taken;
    bool active;
  };
  std::vector<Frame> stack;
  auto is_active = [&]() { return stack.empty() || stack.back().active; };

  struct Source {
    std::string name;
    std::vector<std::string> lines;
    size_t cursor{0};
  };
  std::vector<Source> sources;
  {
    Source root{path, {}, 0};
    std::istringstream stream(text);
    std::string l;
    while (std::getline(stream, l))
      root.lines.push_back(l);
    sources.push_back(std::move(root));
  }

  std::ostringstream output;
  auto trim = [](std::string s) {
    const auto b = s.find_first_not_of(" \t");
    if (b == std::string::npos)
      return std::string();
    const auto e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
  };

  while (!sources.empty()) {
    auto &source = sources.back();
    if (source.cursor >= source.lines.size()) {
      sources.pop_back();
      continue;
    }
    const std::string line = source.lines[source.cursor++];
    const std::string t = trim(line);

    if (t.rfind("#include", 0) == 0 && is_active()) {
      const auto q1 = t.find('"');
      const auto q2 = t.find('"', q1 + 1);
      if (q1 == std::string::npos || q2 == std::string::npos)
        throw std::runtime_error("portable transpiler: bad include in " + source.name + ": " + t);
      std::string include_path = t.substr(q1 + 1, q2 - q1 - 1);
      std::vector<uint8_t> include_data;
      if (vfs_.ReadFile(include_path, include_data) != 0) {
        // Resolve relative to the directory of the including file, as HLSL does.
        const auto slash = source.name.find_last_of('/');
        if (slash != std::string::npos) {
          const std::string relative = source.name.substr(0, slash + 1) + include_path;
          if (vfs_.ReadFile(relative, include_data) == 0)
            include_path = relative;
        }
      }
      if (include_data.empty())
        throw std::runtime_error("portable transpiler: cannot read include: " + include_path + " (from " +
                                 source.name + ")");
      std::string include_text(include_data.begin(), include_data.end());
      Source next{include_path, {}, 0};
      std::istringstream stream(include_text);
      std::string l;
      while (std::getline(stream, l))
        next.lines.push_back(l);
      sources.push_back(std::move(next));
      continue;
    }
    // Only the simple forms used in the shader tree are supported.
    std::function<bool(std::string)> eval_condition = [&](std::string expr) -> bool {
      expr = trim(expr);
      if (expr.rfind("defined", 0) == 0) {
        auto open = expr.find('(');
        auto close = expr.find(')', open == std::string::npos ? 0 : open);
        std::string name;
        if (open != std::string::npos && close != std::string::npos)
          name = expr.substr(open + 1, close - open - 1);
        else
          name = trim(expr.substr(7));
        return defines.count(trim(name)) != 0;
      }
      if (expr.rfind("!defined", 0) == 0)
        return !eval_condition(expr.substr(1));
      if (!expr.empty() && expr[0] == '!')
        return defines.count(trim(expr.substr(1))) == 0;
      return defines.count(expr) != 0;
    };
    if (t.rfind("#ifdef", 0) == 0) {
      const bool cond = defines.count(trim(t.substr(6))) != 0;
      stack.push_back({is_active(), cond, is_active() && cond});
      continue;
    }
    if (t.rfind("#ifndef", 0) == 0) {
      const bool cond = defines.count(trim(t.substr(7))) == 0;
      stack.push_back({is_active(), cond, is_active() && cond});
      continue;
    }
    if (t.rfind("#if", 0) == 0) {
      const bool cond = eval_condition(t.substr(3));
      stack.push_back({is_active(), cond, is_active() && cond});
      continue;
    }
    if (t.rfind("#elif", 0) == 0) {
      if (stack.empty())
        throw std::runtime_error("portable transpiler: #elif without #if in " + source.name);
      auto &frame = stack.back();
      const bool cond = eval_condition(t.substr(5));
      frame.active = frame.parent_active && !frame.branch_taken && cond;
      frame.branch_taken = frame.branch_taken || cond;
      continue;
    }
    if (t.rfind("#else", 0) == 0) {
      if (stack.empty())
        throw std::runtime_error("portable transpiler: #else without #if in " + source.name);
      auto &frame = stack.back();
      frame.active = frame.parent_active && !frame.branch_taken;
      frame.branch_taken = true;
      continue;
    }
    if (t.rfind("#endif", 0) == 0) {
      if (stack.empty())
        throw std::runtime_error("portable transpiler: #endif without #if in " + source.name);
      stack.pop_back();
      continue;
    }
    if (!is_active())
      continue;
    if (t.rfind("#define", 0) == 0) {
      std::string rest = trim(t.substr(7));
      const auto space = rest.find_first_of(" \t");
      const auto paren = rest.find('(');
      if (paren != std::string::npos && (space == std::string::npos || paren < space)) {
        // Function-like macro: join continuation lines and remember the body
        // for token-level expansion in Transform().
        const std::string name = rest.substr(0, paren);
        std::string value = rest;
        while (!value.empty() && value.back() == '\\') {
          value.pop_back();
          if (source.cursor >= source.lines.size())
            break;
          value += trim(source.lines[source.cursor++]);
        }
        defines["#" + name] = "1";
        function_macros[name] = value;
      } else if (space == std::string::npos)
        defines[rest] = "1";
      else {
        const std::string name = rest.substr(0, space);
        const std::string value = trim(rest.substr(space + 1));
        defines[name] = value;
        // Numeric constants (PI, INV_PI, ...) survive as constants in the
        // generated C++ since there is no preprocessor there.
        bool numeric = !value.empty() &&
                       (std::isdigit(static_cast<unsigned char>(value[0])) || value[0] == '-' || value[0] == '+');
        if (numeric) {
          for (char c : value)
            if (!std::isdigit(static_cast<unsigned char>(c)) && c != '.' && c != '-' && c != '+' && c != 'e' &&
                c != 'E' && c != 'f' && c != 'u' && c != 'x' && c != 'X' && c != 'a' && c != 'A' && c != 'b' &&
                c != 'B' && c != 'c' && c != 'C' && c != 'd' && c != 'D' && c != 'F') {
              numeric = false;
              break;
            }
        }
        if (numeric && !defines_.count("@emitted:" + name)) {
          const bool integral = value.find_first_of(".eEfF") == std::string::npos;
          output << "static const " << (integral ? "int" : "float") << " " << name << " = " << value << ";\n";
          defines_["@emitted:" + name] = "1";
        }
        // Type alias defines (`#define Spectrum float3`): expand at rewrite
        // time since there is no preprocessor in the generated C++.
        if (value == "float" || value == "float2" || value == "float3" || value == "float4" || value == "int" ||
            value == "uint" || value == "int2" || value == "int3" || value == "int4" || value == "uint2" ||
            value == "uint3" || value == "uint4" || value == "bool")
          type_aliases_[name] = value;
      }
      continue;
    }
    if (t.rfind("#undef", 0) == 0) {
      defines.erase(trim(t.substr(6)));
      continue;
    }
    if (!t.empty() && t[0] == '#')  // #pragma and friends
      continue;
    output << line << '\n';
  }
  return output.str();
}

std::string Transpiler::Transform(const std::string &source) {
  auto tokens = Tokenize(source);
  std::vector<Token> out;
  out.reserve(tokens.size() * 2);

  // Identifier replacements applied everywhere (they only match HLSL-only
  // names, never C++ keywords).
  static const std::map<std::string, std::string> kIdent = {
      {"saturate", "vsaturate"},   {"clamp", "vclamp"},     {"lerp", "vlerp"},     {"frac", "vfrac"},
      {"floor", "vfloor"},         {"ceil", "vceil"},       {"abs", "vabs"},       {"sign", "vsign"},
      {"length", "vlength"},       {"normalize", "vnormalize"},
      {"reflect", "vreflect"},     {"pow", "vpows"},        {"exp", "vexp"},       {"exp2", "vexp2"},
      {"log", "vlog"},             {"log2", "vlog2"},       {"sqrt", "vsqrt"},     {"rsqrt", "vrsqrt"},
      {"sin", "vsin"},             {"cos", "vcos"},         {"tan", "vtan"},       {"asin", "vasin"},
      {"acos", "vacos"},           {"atan", "vatan"},       {"atan2", "vatan2"},   {"fmod", "vfmod"},
      {"step", "vstep"},           {"isfinite", "visfinite"},
      {"isnan", "visnan"},         {"all", "vall"},         {"any", "vany"},       {"min", "vmin"},
      {"max", "vmax"},             {"trunc", "vtrunc"},     {"round", "vround"},   {"bool2", "int2"},
      {"bool3", "int3"},           {"bool4", "int4"},
      // Resources
      {"data_buffers", "ResourceDataBuffer()"},
      {"sobol_table", "ResourceSobol()"},
      {"camera_data", "ResourceCameraData()"},
      {"instance_metadatas", "ResourceInstanceMetadatas()"},
      {"light_selector_data", "ResourceLightSelector()"},
      {"light_metadatas", "ResourceLightMetadatas()"},
      {"software_instances", "ResourceSoftwareInstances()"},
      {"software_nodes", "ResourceSoftwareNodes()"},
      {"render_settings", "ResourceRenderSettings()"},
      {"accumulated_color", "ResourceAccumulatedColor()"},
      {"accumulated_samples", "ResourceAccumulatedSamples()"},
      // The portable compilation provides a single SampleTexture() helper in
      // hlsl_compat.h; skip the HLSL version from bindings.hlsli.
      {"SampleTexture", "sparkium_portable::SampleTexture"},
  };
  // Identifiers that must be dropped (HLSL keywords without a C++ meaning in
  // the ported subset).
  static const std::set<std::string> kDrop = {"inout", "out",     "in",     "precise", "groupshared",
                                              "unroll", "loop",   "flatten", "branch",  "noinline",
                                              "uniform", "row_major", "column_major"};

  for (size_t i = 0; i < tokens.size(); ++i) {
    const Token &t = tokens[i];
    // CUDA vector-type builtins (float2/3/4, int2/3/4, uint2/3/4, ...) are
    // predefined in the device compilation's global scope even without
    // including <cuda_runtime.h>. A bare reference is therefore ambiguous
    // once `using namespace sparkium_portable;` is active, so these names
    // are always qualified. The "float"/"int" scalar names are unaffected.
    static const std::set<std::string> kCudaBuiltinTypes = {"float2", "float3", "float4", "int2", "int3",
                                                            "int4",   "uint2",  "uint3",  "uint4"};
    auto qualify = [&kCudaBuiltinTypes](const std::string &name) -> std::string {
      if (kCudaBuiltinTypes.count(name))
        return "sparkium_portable::" + name;
      return name;
    };
    // Matrix types (floatNxM) are sparkium_portable-only; qualify them for
    // consistency with the vector types above (they never collide with CUDA
    // builtins, but keeping every type name qualified avoids subtle lookup
    // differences between host and device compilations).
    auto qualify_matrix = [](const std::string &name) -> std::string {
      if (name.size() == 8 && name.rfind("float", 0) == 0 && name[6] == 'x' &&
          std::isdigit(static_cast<unsigned char>(name[5])) && std::isdigit(static_cast<unsigned char>(name[7])))
        return "sparkium_portable::" + name;
      return name;
    };
    if (t.kind == TK_DIRECTIVE)
      continue;  // includes/defines were resolved during expansion
    if (t.kind == TK_IDENT) {
      if (auto macro = function_macros.find(t.text); macro != function_macros.end()) {
        size_t k = i + 1;
        while (k < tokens.size() && tokens[k].kind == TK_WS)
          ++k;
        if (k < tokens.size() && tokens[k].kind == TK_PUNCT && tokens[k].text == "(") {
          int depth = 0;
          size_t j = k;
          for (; j < tokens.size(); ++j) {
            if (tokens[j].kind == TK_PUNCT && tokens[j].text == "(")
              ++depth;
            else if (tokens[j].kind == TK_PUNCT && tokens[j].text == ")") {
              --depth;
              if (depth == 0)
                break;
            }
          }
          if (j < tokens.size()) {
            const std::string args = JoinTokens(tokens, k + 1, j);
            std::string body = macro->second;
            const auto lp = body.find('(');
            const auto rp = body.find(')', lp);
            std::vector<std::string> params;
            if (lp != std::string::npos && rp != std::string::npos) {
              std::string plist = body.substr(lp + 1, rp - lp - 1);
              size_t pos = 0;
              while (pos <= plist.size()) {
                const auto comma = plist.find(',', pos);
                std::string one = TrimCopy(plist.substr(pos, comma == std::string::npos ? std::string::npos
                                                                                    : comma - pos));
                if (!one.empty())
                  params.push_back(one);
                if (comma == std::string::npos)
                  break;
                pos = comma + 1;
              }
              body = TrimCopy(body.substr(rp + 1));
            }
            std::vector<std::string> argv;
            {
              int d = 0;
              size_t start = 0;
              for (size_t q = 0; q <= args.size(); ++q) {
                const char c = q < args.size() ? args[q] : ',';
                if (c == '(' || c == '[')
                  ++d;
                else if (c == ')' || c == ']')
                  --d;
                if ((c == ',' && d == 0) || q == args.size()) {
                  argv.push_back(TrimCopy(args.substr(start, q - start)));
                  start = q + 1;
                }
              }
            }
            for (size_t q = 0; q < params.size() && q < argv.size(); ++q) {
              std::string replaced;
              size_t pos = 0;
              while (pos < body.size()) {
                if (std::isalnum(static_cast<unsigned char>(body[pos])) || body[pos] == '_') {
                  size_t end = pos;
                  while (end < body.size() &&
                         (std::isalnum(static_cast<unsigned char>(body[end])) || body[end] == '_'))
                    ++end;
                  if (body.substr(pos, end - pos) == params[q])
                    replaced += "(" + argv[q] + ")";
                  else
                    replaced += body.substr(pos, end - pos);
                  pos = end;
                } else {
                  replaced += body[pos++];
                }
              }
              body = replaced;
            }
            // Recursively rewrite the expansion so nested intrinsics and
            // macros (clamp, make_float3, ...) are handled as well.
            out.push_back({TK_IDENT, Transform(body)});
            i = j;
            continue;
          }
        }
        out.push_back(t);
        continue;
      }
      if (auto alias = type_aliases_.find(t.text); alias != type_aliases_.end()) {
        // Alias values are bare HLSL type names (see the type_aliases_
        // registration); qualify the CUDA-conflicting vector names as below.
        out.push_back({TK_IDENT, qualify(alias->second)});
        continue;
      }
      if (kCudaBuiltinTypes.count(t.text)) {
        out.push_back({TK_IDENT, "sparkium_portable::" + t.text});
        continue;
      }
      // make_floatN helpers from bsdf/principled_util.hlsli are skipped in
      // the portable compilation (they collide with CUDA builtins); the
      // call sites map onto the floatN constructors, which accept the same
      // 1- and 3-argument forms.
      if (t.text == "make_float2" || t.text == "make_float3" || t.text == "make_float4") {
        out.push_back({TK_IDENT, "sparkium_portable::" + t.text.substr(5)});
        continue;
      }
      if (auto it = kIdent.find(t.text); it != kIdent.end()) {
        // Resource expansions like ResourceAccumulatedSamples() must not be
        // rewritten when used as member names (render_settings.accumulated_samples).
        bool member_access = false;
        if (!it->second.empty() && it->second.back() == ')') {
          for (size_t q = out.size(); q-- > 0;) {
            if (out[q].kind == TK_WS)
              continue;
            member_access = out[q].kind == TK_PUNCT && out[q].text == ".";
            break;
          }
        }
        if (member_access)
          out.push_back(t);
        else
          out.push_back({TK_IDENT, it->second});
        continue;
      }
      if (t.text == "static") {
        // HLSL allows in-class initialized `static const float` members; C++
        // requires constexpr there.
        size_t k = i + 1;
        while (k < tokens.size() && tokens[k].kind == TK_WS)
          ++k;
        if (k < tokens.size() && tokens[k].text == "const") {
          size_t m = k + 1;
          while (m < tokens.size() && tokens[m].kind == TK_WS)
            ++m;
          if (m < tokens.size() &&
              (tokens[m].text == "float" || tokens[m].text == "int" || tokens[m].text == "uint")) {
            out.push_back({TK_IDENT, "static constexpr"});
            i = k;
            continue;
          }
        }
        out.push_back(t);
        continue;
      }
      if (t.text == "inout" || t.text == "out") {
        // Resolve type aliases before looking for the parameter name.
        size_t tk = i + 1;
        while (tk < tokens.size() && tokens[tk].kind == TK_WS)
          ++tk;
        if (tk < tokens.size() && tokens[tk].kind == TK_IDENT) {
          if (auto alias = type_aliases_.find(tokens[tk].text); alias != type_aliases_.end()) {
            out.push_back({TK_IDENT, qualify(alias->second)});
            out.push_back({TK_PUNCT, "&"});
            // find the name
            size_t m = tk + 1;
            while (m < tokens.size() && tokens[m].kind == TK_WS)
              ++m;
            if (m < tokens.size() && tokens[m].kind == TK_IDENT) {
              out.push_back(tokens[m]);
              i = m;
            } else {
              i = tk;
            }
            continue;
          }
        }
        // HLSL inout/out parameters become C++ references: mark the
        // following declared identifier with '&'.
        size_t k = i + 1;
        while (k < tokens.size() && tokens[k].kind == TK_WS)
          ++k;
        // tokens[k] is the type; find the parameter name (last ident before
        // ',' ')' or '=').
        size_t m = k;
        size_t name_pos = std::string::npos;
        int angle = 0;
        for (; m < tokens.size(); ++m) {
          if (tokens[m].kind == TK_PUNCT) {
            if (tokens[m].text == "<")
              ++angle;
            else if (tokens[m].text == ">")
              --angle;
            else if ((tokens[m].text == "," || tokens[m].text == ")" || tokens[m].text == "=") && angle == 0)
              break;
          }
          if (tokens[m].kind == TK_IDENT && angle == 0)
            name_pos = m;
        }
        out.push_back({TK_IDENT, qualify(tokens[k].text)});  // the type
        if (name_pos != std::string::npos) {
          // Emit tokens between type and name (e.g. template args) verbatim.
          for (size_t q = k + 1; q < name_pos; ++q)
            out.push_back(tokens[q]);
          out.push_back({TK_PUNCT, "&"});
          out.push_back(tokens[name_pos]);
          i = name_pos;
        } else {
          i = k;
        }
        continue;
      }
      if (kDrop.count(t.text)) {
        // Preserve spacing; dropping the token entirely is fine because HLSL
        // keywords here sit between other tokens with whitespace.
        continue;
      }
      // Strip HLSL semantics and attributes: handled at the punct level
      // below.
      out.push_back(t);
      continue;
    }
    if (t.kind == TK_PUNCT) {
      // Drop [numthreads(...)], [shader("...")], [vk::location(N)] and
      // : register(...) / : SEMANTIC annotations.
      if (t.text == "[") {
        // Find matching ].
        int depth = 1;
        size_t j = i + 1;
        while (j < tokens.size() && depth > 0) {
          if (tokens[j].kind == TK_PUNCT && tokens[j].text == "[")
            ++depth;
          if (tokens[j].kind == TK_PUNCT && tokens[j].text == "]")
            --depth;
          ++j;
        }
        // Look at the inner content to decide whether this is an attribute.
        size_t k = i + 1;
        while (k < j && tokens[k].kind == TK_WS)
          ++k;
        bool attribute = false;
        if (k < j && tokens[k].kind == TK_IDENT) {
          const std::string &name = tokens[k].text;
          attribute = name == "numthreads" || name == "shader" || name == "vk";
        }
        if (attribute) {
          i = j - 1;
          continue;
        }
        out.push_back(t);
        continue;
      }
      if (t.text == ":") {
        // Skip ": register(x, spaceN)" and ": SEMANTIC" annotations. Keep
        // "::" (tokenized separately) and labels/ternaries: the shaders use
        // ':' after declarations only for registers/semantics.
        size_t k = i + 1;
        while (k < tokens.size() && tokens[k].kind == TK_WS)
          ++k;
        if (k < tokens.size() && tokens[k].kind == TK_IDENT) {
          const std::string &name = tokens[k].text;
          if (name == "register") {
            // Skip to the matching close paren.
            size_t j = k + 1;
            while (j < tokens.size() && !(tokens[j].kind == TK_PUNCT && tokens[j].text == "("))
              ++j;
            int depth = 0;
            while (j < tokens.size()) {
              if (tokens[j].kind == TK_PUNCT && tokens[j].text == "(")
                ++depth;
              else if (tokens[j].kind == TK_PUNCT && tokens[j].text == ")") {
                --depth;
                if (depth == 0)
                  break;
              }
              ++j;
            }
            i = j;
            continue;
          }
          if (name.rfind("SV_", 0) == 0 || name.rfind("TEXCOORD", 0) == 0 || name.rfind("COLOR", 0) == 0 ||
              name.rfind("TARGET", 0) == 0) {
            i = k;
            continue;
          }
        }
        out.push_back(t);
        continue;
      }
      if (t.text == "." && IsSwizzle(tokens, i) && tokens[i + 1].text.size() > 1) {
        // Assignment to a multi-component swizzle: emit component stores.
        {
          size_t k = i + 2;
          while (k < tokens.size() && tokens[k].kind == TK_WS)
            ++k;
          if (k < tokens.size() && tokens[k].kind == TK_PUNCT &&
              (tokens[k].text == "=" || tokens[k].text == "+=" || tokens[k].text == "-=" ||
               tokens[k].text == "*=" || tokens[k].text == "/=")) {
            const std::string field = tokens[i + 1].text;
            const std::string op = tokens[k].text;
            const size_t start = ExpressionStart(out, out.size());
            const std::string base = JoinTokens(out, start, out.size());
            out.erase(out.begin() + start, out.end());
            // Find the RHS up to the terminating ';' at top level.
            int depth = 0;
            size_t j = k + 1;
            for (; j < tokens.size(); ++j) {
              if (tokens[j].kind == TK_PUNCT) {
                if (tokens[j].text == "(" || tokens[j].text == "[")
                  ++depth;
                else if (tokens[j].text == ")" || tokens[j].text == "]")
                  --depth;
                else if (tokens[j].text == ";" && depth == 0)
                  break;
              }
            }
            const std::string rhs = JoinTokens(tokens, k + 1, j);
            static const std::map<char, int> kIndex{{'x', 0}, {'y', 1}, {'z', 2}, {'w', 3},
                                                    {'r', 0}, {'g', 1}, {'b', 2}, {'a', 3}};
            std::string stmt = "SparkiumSwizzleAssign(" + base + ", " + rhs + ", " +
                               std::to_string(field.size()) + ", ";
            for (size_t q = 0; q < field.size(); ++q) {
              if (q)
                stmt += ", ";
              stmt += std::to_string(kIndex.at(field[q]));
            }
            stmt += ");";
            if (op == "=") {
              out.push_back({TK_IDENT, stmt});
            } else {
              // Compound assignment: read the swizzle, apply element-wise,
              // write back through SparkiumSwizzleAssign (handles scalars).
              const std::string make = "float" + std::to_string(field.size()) + "(";
              std::string read = make;
              for (size_t q = 0; q < field.size(); ++q) {
                if (q)
                  read += ",";
                read += "(" + base + ")[" + std::to_string(kIndex.at(field[q])) + "]";
              }
              read += ")";
              const std::string plain = op.substr(0, 1);
              std::string stmt = "SparkiumSwizzleAssign(" + base + ", (" + read + ") " + plain + " (" + rhs + "), " +
                                 std::to_string(field.size()) + ", ";
              for (size_t q = 0; q < field.size(); ++q) {
                if (q)
                  stmt += ", ";
                stmt += std::to_string(kIndex.at(field[q]));
              }
              stmt += ");";
              out.push_back({TK_IDENT, stmt});
            }
            i = j;  // consume through ';'
            continue;
          }
        }
        // Multi-component swizzle: find the base expression and rebuild it
        // as a vector construction.
        const std::string field = tokens[i + 1].text;
        const size_t start = ExpressionStart(out, out.size());
        const std::string base = JoinTokens(out, start, out.size());
        out.erase(out.begin() + start, out.end());
        static const std::map<char, int> kIndex{{'x', 0}, {'y', 1}, {'z', 2}, {'w', 3},
                                                {'r', 0}, {'g', 1}, {'b', 2}, {'a', 3}};
        std::string make = "float" + std::to_string(field.size()) + "(";
        for (size_t q = 0; q < field.size(); ++q) {
          if (q)
            make += ",";
          make += "(" + base + ")[" + std::to_string(kIndex.at(field[q])) + "]";
        }
        make += ")";
        out.push_back({TK_IDENT, make});
        ++i;  // consume the swizzle identifier
        continue;
      }
      if (t.text == "(") {
        // HLSL zero-cast `(Type)0` value-initializes structs.
        size_t k = i + 1;
        while (k < tokens.size() && tokens[k].kind == TK_WS)
          ++k;
        if (k < tokens.size() && tokens[k].kind == TK_IDENT) {
          size_t m = k + 1;
          while (m < tokens.size() && tokens[m].kind == TK_WS)
            ++m;
          if (m < tokens.size() && tokens[m].kind == TK_PUNCT && tokens[m].text == ")") {
            size_t n = m + 1;
            while (n < tokens.size() && tokens[n].kind == TK_WS)
              ++n;
            if (n < tokens.size() && tokens[n].kind == TK_NUMBER && tokens[n].text == "0") {
              const std::string &type = tokens[k].text;
              if (type != "int" && type != "uint" && type != "float" && type != "double" &&
                  type.rfind("float", 0) != 0 && type.rfind("uint", 0) != 0 && type.rfind("int", 0) != 0) {
                out.push_back(t);
                out.push_back(tokens[k]);
                out.push_back(tokens[m]);
                out.push_back({TK_IDENT, "{}"});
                i = n;  // consume the 0
                continue;
              }
            }
          }
        }
        out.push_back(t);
        continue;
      }
      if (t.text == "{") {
        // HLSL class members are public by default in these shaders.
        size_t k = out.size();
        while (k > 0 && (out[k - 1].kind == TK_WS))
          --k;
        bool is_class = false;
        for (size_t q = k; q-- > 0;) {
          if (out[q].kind == TK_WS)
            continue;
          if (out[q].kind == TK_IDENT) {
            if (out[q].text == "class") {
              is_class = true;
              break;
            }
            if (out[q].text == "struct")
              break;
            continue;  // skip the class name
          }
          break;
        }
        out.push_back(t);
        if (is_class)
          out.push_back({TK_IDENT, " public:"});
        continue;
      }
      out.push_back(t);
      continue;
    }
    out.push_back(t);
  }

  std::string result = JoinTokens(out, 0, out.size());
  result = AnnotateDeviceFunctions(result);
  result = RewriteAsFloatVectorCasts(result);
  // Code synthesized during the rewrite (multi-component swizzle expansion,
  // function-macro bodies) contains bare floatN(...)/intN(...)/uintN(...)
  // constructor expressions. Qualify them for CUDA, where the global builtin
  // vector types would otherwise make the name ambiguous (see the
  // kCudaBuiltinTypes comment above). Already-qualified names are skipped.
  static const char *kBuiltinNames[] = {"float2", "float3", "float4", "int2", "int3", "int4",
                                        "uint2",  "uint3",  "uint4"};
  for (const char *name : kBuiltinNames) {
    const std::string needle = std::string(name) + "(";
    const std::string qualified = std::string("sparkium_portable::") + name + "(";
    size_t pos = 0;
    while ((pos = result.find(needle, pos)) != std::string::npos) {
      if (pos >= 19 && result.compare(pos - 19, 19, "sparkium_portable::") == 0) {
        pos += qualified.size();
        continue;
      }
      // Skip identifier continuations: LoadUint2( must not become
      // LoadUsparkium_portable::int2(. Only genuine constructor expressions
      // (preceded by punctuation, whitespace, or start of file) qualify.
      if (pos > 0) {
        const char prev = result[pos - 1];
        if (std::isalnum(static_cast<unsigned char>(prev)) || prev == '_') {
          pos += needle.size();
          continue;
        }
      }
      result.replace(pos, needle.size(), qualified);
      pos += qualified.size();
    }
  }
  // Normalize float suffixes: HLSL `1.0f` is valid C++, `1.0F` too; nothing
  // to do. Remove duplicate blank lines.
  return result;
}

}  // namespace sparkium::portable
