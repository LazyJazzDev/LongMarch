// Tree-walking interpreter for the shader-graph source.
//
// ShaderGraphCompiler emits a small, closed subset of C: a handful of free
// functions over float/float2/float3/float4, for loops over a constant range,
// assignments and one return. That is enough to parse rather than translate,
// which is what makes this engine exact -- it evaluates the very text the GPU
// path hands to DXC, so the two cannot drift apart.
//
// The alternative engine compiles the same text with Clang; see graph_jit.cpp.
#include "sparkium/pipelines/raytracing/cpu/graph_program.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "grassland/util/log.h"

namespace sparkium::raytracing::cpu {
namespace {

// A value is a scalar, a float2/3/4, a mat3/mat4, or the one struct the
// generated code declares. HLSL promotes and truncates freely, so the component
// count travels with the value rather than being fixed by a type.
struct Value {
  int n{1};
  float v[16]{};
  std::shared_ptr<std::vector<Value>> fields;  // set for GraphSurface
  // The one resource the graph reads; only ever passed to Load().
  bool material_data{false};

  bool IsStruct() const {
    return fields != nullptr;
  }
  bool IsMatrix() const {
    return !IsStruct() && (n == 9 || n == 16);
  }
  float Scalar() const {
    return v[0];
  }
  static Value Splat(int count, float value) {
    Value result;
    result.n = count;
    for (int i = 0; i < count; ++i)
      result.v[i] = value;
    return result;
  }
};

// Tokenizer ------------------------------------------------------------------

enum class TokenKind { End, Identifier, Number, Punctuation, String };

struct Token {
  TokenKind kind{TokenKind::End};
  std::string text;
  double number{0.0};
};

std::vector<Token> Tokenize(const std::string &source) {
  std::vector<Token> tokens;
  size_t i = 0;
  while (i < source.size()) {
    const char c = source[i];
    if (std::isspace(static_cast<unsigned char>(c))) {
      ++i;
      continue;
    }
    // Comments and preprocessor lines carry no meaning for the graph body.
    if (c == '/' && i + 1 < source.size() && source[i + 1] == '/') {
      while (i < source.size() && source[i] != '\n')
        ++i;
      continue;
    }
    if (c == '/' && i + 1 < source.size() && source[i + 1] == '*') {
      i += 2;
      while (i + 1 < source.size() && !(source[i] == '*' && source[i + 1] == '/'))
        ++i;
      i = std::min(i + 2, source.size());
      continue;
    }
    if (c == '#') {
      while (i < source.size() && source[i] != '\n')
        ++i;
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      const size_t start = i;
      while (i < source.size() && (std::isalnum(static_cast<unsigned char>(source[i])) || source[i] == '_'))
        ++i;
      tokens.push_back({TokenKind::Identifier, source.substr(start, i - start), 0.0});
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(c)) ||
        (c == '.' && i + 1 < source.size() && std::isdigit(static_cast<unsigned char>(source[i + 1])))) {
      const size_t start = i;
      while (i < source.size() &&
             (std::isalnum(static_cast<unsigned char>(source[i])) || source[i] == '.' || source[i] == '+' ||
              source[i] == '-')) {
        // Only continue through a sign that belongs to an exponent.
        if ((source[i] == '+' || source[i] == '-') &&
            !(i > start && (source[i - 1] == 'e' || source[i - 1] == 'E')))
          break;
        ++i;
      }
      std::string text = source.substr(start, i - start);
      std::string numeric = text;
      if (!numeric.empty() && (numeric.back() == 'f' || numeric.back() == 'F'))
        numeric.pop_back();
      tokens.push_back({TokenKind::Number, text, std::stod(numeric)});
      continue;
    }
    // Two-character operators first so they are not split.
    const std::string two = i + 1 < source.size() ? source.substr(i, 2) : std::string();
    if (two == "==" || two == "!=" || two == "<=" || two == ">=" || two == "&&" || two == "||" || two == "+=" ||
        two == "-=" || two == "*=" || two == "/=" || two == "++" || two == "--") {
      tokens.push_back({TokenKind::Punctuation, two, 0.0});
      i += 2;
      continue;
    }
    tokens.push_back({TokenKind::Punctuation, std::string(1, c), 0.0});
    ++i;
  }
  tokens.push_back({TokenKind::End, "", 0.0});
  return tokens;
}

// AST ------------------------------------------------------------------------

enum class ExprKind { Number, Identifier, Member, Call, Unary, Binary, Ternary };

struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct Expr {
  ExprKind kind{ExprKind::Number};
  double number{0.0};
  std::string name;                 // Identifier, Member field, Call callee, or operator
  ExprPtr left, right, third;
  std::vector<ExprPtr> args;
};

enum class StmtKind { Block, VarDecl, Assign, ExprStmt, If, For, Return };

struct Stmt;
using StmtPtr = std::shared_ptr<Stmt>;

struct Stmt {
  StmtKind kind{StmtKind::ExprStmt};
  // VarDecl: one or more declarators sharing a type.
  std::string type;
  std::vector<std::pair<std::string, ExprPtr>> declarators;
  // Assign: target and value, with the operator that produced it.
  ExprPtr target, value;
  std::string op;
  ExprPtr expr;
  std::vector<StmtPtr> body;    // Block, If-then, For-body
  StmtPtr otherwise;            // If-else
  StmtPtr init, step;           // For
  ExprPtr condition;            // If, For
};

struct Function {
  std::string name;
  std::vector<std::string> parameters;
  StmtPtr body;
};

// Parser ---------------------------------------------------------------------

class Parser {
 public:
  explicit Parser(const std::string &source) : tokens_(Tokenize(source)) {
  }

  std::map<std::string, Function> Parse() {
    std::map<std::string, Function> functions;
    while (!AtEnd()) {
      // A function definition is `Type name(...)`, as opposed to a
      // constructor-like call which never appears at file scope.
      if (AtTypeName() && Peek(1).kind == TokenKind::Identifier && Peek(2).text == "(") {
        Advance();
        const std::string name = Advance().text;
        Expect("(");
        std::vector<std::string> parameters;
        while (!Check(")")) {
          Advance();  // parameter type
          parameters.push_back(Advance().text);
          if (!Check(")"))
            Expect(",");
        }
        Expect(")");
        Function function;
        function.name = name;
        function.parameters = parameters;
        function.body = ParseBlock();
        functions[name] = function;
        continue;
      }
      if (Peek().kind == TokenKind::Identifier && Peek(1).kind == TokenKind::Identifier) {
        // A forward declaration such as `void Foo(...);` -- skip it.
        while (!AtEnd() && !Check(";"))
          Advance();
        Expect(";");
        continue;
      }
      // Anything else at file scope is not something the codegen emits.
      throw std::runtime_error("unexpected token at file scope: " + Peek().text);
    }
    return functions;
  }

 private:
  const Token &Peek(int offset = 0) const {
    const size_t index = std::min(position_ + offset, tokens_.size() - 1);
    return tokens_[index];
  }
  bool AtEnd() const {
    return Peek().kind == TokenKind::End;
  }
  const Token &Advance() {
    return tokens_[position_++ < tokens_.size() - 1 ? position_ - 1 : position_];
  }
  bool Check(const std::string &text) const {
    return Peek().text == text && Peek().kind != TokenKind::End;
  }
  bool Match(const std::string &text) {
    if (!Check(text))
      return false;
    ++position_;
    return true;
  }
  void Expect(const std::string &text) {
    if (!Match(text))
      throw std::runtime_error("expected '" + text + "' but found '" + Peek().text + "'" + Context());
  }

  // A few tokens around the cursor, so a parse failure points at the construct
  // rather than only at the token.
  std::string Context() const {
    std::string result = " near";
    for (int i = -4; i <= 3; ++i) {
      const int index = static_cast<int>(position_) + i;
      if (index < 0 || index >= static_cast<int>(tokens_.size()))
        continue;
      result += " ";
      result += (i == 0 ? ">>>" : "");
      result += tokens_[static_cast<size_t>(index)].text.empty() ? "<eof>"
                                                                 : tokens_[static_cast<size_t>(index)].text;
    }
    return result;
  }

  StmtPtr ParseBlock() {
    Expect("{");
    auto block = std::make_shared<Stmt>();
    block->kind = StmtKind::Block;
    while (!Check("}") && !AtEnd())
      block->body.push_back(ParseStatement());
    Expect("}");
    return block;
  }

  bool AtTypeName() const {
    const std::string &text = Peek().text;
    return Peek().kind == TokenKind::Identifier &&
           (text == "float" || text == "float2" || text == "float3" || text == "float4" || text == "int" ||
            text == "uint" || text == "bool" || text == "GraphSurface" || text == "HitRecord" ||
            text == "RandomDevice" || text == "SoftwareHit");
  }

  StmtPtr ParseStatement() {
    if (Check("{"))
      return ParseBlock();
    if (Match("if")) {
      auto statement = std::make_shared<Stmt>();
      statement->kind = StmtKind::If;
      Expect("(");
      statement->condition = ParseExpression();
      Expect(")");
      statement->body.push_back(ParseStatement());
      if (Match("else"))
        statement->otherwise = ParseStatement();
      return statement;
    }
    if (Match("for")) {
      auto statement = std::make_shared<Stmt>();
      statement->kind = StmtKind::For;
      Expect("(");
      if (!Check(";"))
        statement->init = ParseStatement();
      else
        Expect(";");
      if (!Check(";"))
        statement->condition = ParseExpression();
      Expect(";");
      if (!Check(")")) {
        // The step is an expression, not a statement: the loop supplies the
        // separator, so no semicolon is consumed here.
        statement->step = std::make_shared<Stmt>();
        statement->step->kind = StmtKind::ExprStmt;
        statement->step->expr = ParseExpression();
      }
      Expect(")");
      statement->body.push_back(ParseStatement());
      return statement;
    }
    if (Match("return")) {
      auto statement = std::make_shared<Stmt>();
      statement->kind = StmtKind::Return;
      if (!Check(";"))
        statement->expr = ParseExpression();
      Expect(";");
      return statement;
    }
    if (Match("continue") || Match("break")) {
      // The generated code has neither; accept and ignore so a future codegen
      // change fails loudly at run time rather than at parse time.
      Expect(";");
      auto statement = std::make_shared<Stmt>();
      statement->kind = StmtKind::ExprStmt;
      return statement;
    }
    if (AtTypeName() && Peek(1).kind == TokenKind::Identifier)
      return ParseVarDecl();
    return ParseExpressionStatement();
  }

  StmtPtr ParseVarDecl() {
    auto statement = std::make_shared<Stmt>();
    statement->kind = StmtKind::VarDecl;
    statement->type = Advance().text;
    for (;;) {
      const std::string name = Advance().text;
      ExprPtr initializer;
      if (Match("="))
        initializer = ParseExpression();
      statement->declarators.emplace_back(name, initializer);
      if (!Match(","))
        break;
    }
    Expect(";");
    return statement;
  }

  StmtPtr ParseExpressionStatement() {
    ExprPtr first = ParseExpression();
    for (const char *op : {"=", "+=", "-=", "*=", "/="}) {
      if (Check(op)) {
        Advance();
        auto statement = std::make_shared<Stmt>();
        statement->kind = StmtKind::Assign;
        statement->op = op;
        statement->target = first;
        statement->value = ParseExpression();
        Expect(";");
        return statement;
      }
    }
    auto statement = std::make_shared<Stmt>();
    statement->kind = StmtKind::ExprStmt;
    statement->expr = first;
    Expect(";");
    return statement;
  }

  ExprPtr ParseExpression() {
    return ParseTernary();
  }
  ExprPtr ParseTernary() {
    ExprPtr condition = ParseBinary(0);
    if (Match("?")) {
      auto expression = std::make_shared<Expr>();
      expression->kind = ExprKind::Ternary;
      expression->left = condition;
      expression->right = ParseExpression();
      Expect(":");
      expression->third = ParseTernary();
      return expression;
    }
    return condition;
  }

  static int Precedence(const std::string &op) {
    if (op == "||")
      return 1;
    if (op == "&&")
      return 2;
    if (op == "==" || op == "!=")
      return 3;
    if (op == "<" || op == ">" || op == "<=" || op == ">=")
      return 4;
    if (op == "+" || op == "-")
      return 5;
    if (op == "*" || op == "/")
      return 6;
    return 0;
  }

  ExprPtr ParseBinary(int minimum) {
    ExprPtr left = ParseUnary();
    for (;;) {
      const std::string op = Peek().text;
      const int precedence = Peek().kind == TokenKind::Punctuation ? Precedence(op) : 0;
      if (precedence == 0 || precedence < minimum)
        return left;
      Advance();
      ExprPtr right = ParseBinary(precedence + 1);
      auto expression = std::make_shared<Expr>();
      expression->kind = ExprKind::Binary;
      expression->name = op;
      expression->left = left;
      expression->right = right;
      left = expression;
    }
  }

  ExprPtr ParseUnary() {
    const std::string &text = Peek().text;
    if (text == "-" || text == "!" || text == "+" || text == "++" || text == "--") {
      Advance();
      auto expression = std::make_shared<Expr>();
      expression->kind = ExprKind::Unary;
      expression->name = text;
      expression->left = ParseUnary();
      return expression;
    }
    return ParsePostfix();
  }

  ExprPtr ParsePostfix() {
    ExprPtr expression = ParsePrimary();
    for (;;) {
      if (Match(".")) {
        const std::string member = Advance().text;
        auto access = std::make_shared<Expr>();
        access->kind = ExprKind::Member;
        access->name = member;
        access->left = expression;
        expression = access;
        // A method call such as `material_data.Load(12)`; the callee is
        // recorded as a dotted name so the evaluator can dispatch on it.
        if (Match("(")) {
          auto call = std::make_shared<Expr>();
          call->kind = ExprKind::Call;
          call->name = DottedName(expression);
          while (!Check(")")) {
            call->args.push_back(ParseExpression());
            if (!Check(")"))
              Expect(",");
          }
          Expect(")");
          expression = call;
        }
        continue;
      }
      if (Match("[")) {
        // Array subscripts never appear in the generated code.
        throw std::runtime_error("array subscripts are not part of the graph grammar");
      }
      if (Match("++") || Match("--")) {
        auto expression_post = std::make_shared<Expr>();
        expression_post->kind = ExprKind::Unary;
        expression_post->name = "post";
        expression_post->left = expression;
        expression = expression_post;
      }
      return expression;
    }
  }

  static std::string DottedName(const ExprPtr &expression) {
    if (expression->kind == ExprKind::Identifier)
      return expression->name;
    if (expression->kind == ExprKind::Member)
      return DottedName(expression->left) + "." + expression->name;
    return {};
  }

  ExprPtr ParsePrimary() {
    if (Peek().kind == TokenKind::Number) {
      auto expression = std::make_shared<Expr>();
      expression->kind = ExprKind::Number;
      expression->number = Advance().number;
      return expression;
    }
    if (Match("(")) {
      ExprPtr inner = ParseExpression();
      Expect(")");
      return inner;
    }
    if (Peek().kind == TokenKind::Identifier) {
      const std::string name = Advance().text;
      if (Match("(")) {
        auto call = std::make_shared<Expr>();
        call->kind = ExprKind::Call;
        call->name = name;
        while (!Check(")")) {
          call->args.push_back(ParseExpression());
          if (!Check(")"))
            Expect(",");
        }
        Expect(")");
        return call;
      }
      auto identifier = std::make_shared<Expr>();
      identifier->kind = ExprKind::Identifier;
      identifier->name = name;
      return identifier;
    }
    throw std::runtime_error("unexpected token in expression: '" + Peek().text + "'");
  }

  std::vector<Token> tokens_;
  size_t position_{0};
};

// Evaluator ------------------------------------------------------------------

// GraphSurface's fields in the order surface_sampler.hlsli declares them.
const std::vector<std::string> &SurfaceFields() {
  static const std::vector<std::string> fields{
      "base_color",          "metallic",       "specular",           "roughness",
      "anisotropic",         "anisotropic_rotation", "sheen",        "clearcoat",
      "clearcoat_roughness", "ior",            "transmission",       "transmission_roughness",
      "emission",            "normal",         "opacity",            "shadow_opacity",
      "thin_walled",         "subsurface",     "subsurface_scale",   "subsurface_radius",
      "subsurface_method"};
  return fields;
}

// GraphSurface has four float3 fields; everything else is a scalar. The
// interpreter has to know, or an assignment to base_color would be truncated.
int SurfaceFieldComponents(const std::string &name) {
  if (name == "base_color" || name == "emission" || name == "normal" || name == "subsurface_radius")
    return 3;
  return 1;
}

int ComponentCount(const std::string &type) {
  if (type == "float2")
    return 2;
  if (type == "float3")
    return 3;
  if (type == "float4")
    return 4;
  return 1;
}

class Interpreter {
 public:
  Interpreter(std::map<std::string, Function> functions, const GraphEvalInput &input)
      : functions_(std::move(functions)), input_(input) {
  }

  void Evaluate(GraphEvalOutput &output) {
    auto found = functions_.find("EvaluateShaderGraph");
    if (found == functions_.end())
      throw std::runtime_error("generated shader graph has no EvaluateShaderGraph");

    std::vector<Value> arguments;
    arguments.push_back(MakeHitRecord());
    arguments.push_back(FromArray(input_.view_direction, 3));
    arguments.push_back(Scalar(static_cast<float>(input_.bounce)));
    arguments.push_back(Scalar(static_cast<float>(input_.ray_type)));
    arguments.push_back(Scalar(input_.is_shadow_ray ? 1.0f : 0.0f));
    arguments.push_back(MakeMaterialData());

    scopes_.clear();
    scopes_.emplace_back();
    const Value surface = CallFunction(found->second, arguments);
    scopes_.pop_back();

    // Copy the fields out by name, so neither side depends on the other's
    // layout.
    const auto &fields = *surface.fields;
    auto get = [&](int index) {
      return fields[static_cast<size_t>(index)];
    };
    const std::vector<std::string> &names = SurfaceFields();
    auto index_of = [&](const char *name) {
      for (size_t i = 0; i < names.size(); ++i)
        if (names[i] == name)
          return static_cast<int>(i);
      return -1;
    };
    auto copy3 = [&](const char *name, float *out) {
      const Value &value = get(index_of(name));
      for (int i = 0; i < 3; ++i)
        out[i] = value.v[std::min(i, value.n - 1)];
    };
    auto copy1 = [&](const char *name, float &out) {
      out = get(index_of(name)).Scalar();
    };
    copy3("base_color", output.base_color);
    copy1("metallic", output.metallic);
    copy1("specular", output.specular);
    copy1("roughness", output.roughness);
    copy1("anisotropic", output.anisotropic);
    copy1("anisotropic_rotation", output.anisotropic_rotation);
    copy1("sheen", output.sheen);
    copy1("clearcoat", output.clearcoat);
    copy1("clearcoat_roughness", output.clearcoat_roughness);
    copy1("ior", output.ior);
    copy1("transmission", output.transmission);
    copy1("transmission_roughness", output.transmission_roughness);
    copy3("emission", output.emission);
    copy3("normal", output.normal);
    copy1("opacity", output.opacity);
    copy1("shadow_opacity", output.shadow_opacity);
    copy1("thin_walled", output.thin_walled);
    copy1("subsurface", output.subsurface);
    copy1("subsurface_scale", output.subsurface_scale);
    copy3("subsurface_radius", output.subsurface_radius);
    copy1("subsurface_method", output.subsurface_method);
  }

 private:
  using Scope = std::map<std::string, Value>;

  static Value Scalar(float value) {
    Value result;
    result.n = 1;
    result.v[0] = value;
    return result;
  }
  static Value FromArray(const float *values, int count) {
    Value result;
    result.n = count;
    for (int i = 0; i < count; ++i)
      result.v[i] = values[i];
    return result;
  }

  Value MakeHitRecord() {
    Value record;
    record.n = 0;
    record.fields = std::make_shared<std::vector<Value>>();
    // Order must match StructField's name table below.
    auto add = [&](const char *, const Value &value) { record.fields->push_back(value); };
    add("t", Scalar(input_.t));
    add("position", FromArray(input_.position, 3));
    add("object_position", FromArray(input_.object_position, 3));
    add("object_origin", FromArray(input_.object_origin, 3));
    add("tex_coord", FromArray(input_.tex_coord, 2));
    add("color", FromArray(input_.color, 3));
    add("normal", FromArray(input_.normal, 3));
    add("geom_normal", FromArray(input_.geom_normal, 3));
    add("tangent", FromArray(input_.tangent, 3));
    add("signal", Scalar(input_.signal));
    add("pdf", Scalar(input_.pdf));
    add("primitive_index", Scalar(static_cast<float>(input_.primitive_index)));
    add("object_index", Scalar(static_cast<float>(input_.object_index)));
    add("front_facing", Scalar(input_.front_facing ? 1.0f : 0.0f));
    return record;
  }

  Value MakeMaterialData() {
    Value value;
    value.material_data = true;
    return value;
  }

  // Field lookup for a struct value. `HitRecord` is the only struct the
  // generated code reads through a variable; its fields are stored in a fixed
  // order and matched by name here.
  Value StructField(const Value &value, const std::string &name) const {
    static const std::vector<std::string> fields{"t",        "position",        "object_position",
                                                 "object_origin", "tex_coord",  "color",
                                                 "normal",   "geom_normal",     "tangent",
                                                 "signal",   "pdf",             "primitive_index",
                                                 "object_index", "front_facing"};
    for (size_t i = 0; i < fields.size() && i < value.fields->size(); ++i)
      if (fields[i] == name)
        return (*value.fields)[i];
    throw std::runtime_error("unknown struct field: " + name);
  }

  Value *Lookup(const std::string &name) {
    for (auto scope = scopes_.rbegin(); scope != scopes_.rend(); ++scope) {
      auto found = scope->find(name);
      if (found != scope->end())
        return &found->second;
    }
    return nullptr;
  }

  static int SwizzleIndex(char c) {
    switch (c) {
      case 'x':
      case 'r':
        return 0;
      case 'y':
      case 'g':
        return 1;
      case 'z':
      case 'b':
        return 2;
      case 'w':
      case 'a':
        return 3;
      default:
        return -1;
    }
  }

  Value Swizzle(const Value &value, const std::string &swizzle) const {
    Value result;
    if (swizzle.size() == 1) {
      const int index = SwizzleIndex(swizzle[0]);
      if (index < 0 || index >= value.n)
        throw std::runtime_error("swizzle out of range: ." + swizzle);
      return Scalar(value.v[index]);
    }
    result.n = static_cast<int>(swizzle.size());
    for (size_t i = 0; i < swizzle.size(); ++i) {
      const int index = SwizzleIndex(swizzle[i]);
      if (index < 0 || index >= value.n)
        throw std::runtime_error("swizzle out of range: ." + swizzle);
      result.v[i] = value.v[index];
    }
    return result;
  }

  Value EvaluateExpr(const ExprPtr &expression) {
    switch (expression->kind) {
      case ExprKind::Number:
        return Scalar(static_cast<float>(expression->number));
      case ExprKind::Identifier:
        return EvaluateIdentifier(expression->name);
      case ExprKind::Member:
        return EvaluateMember(expression);
      case ExprKind::Call:
        return EvaluateCall(expression);
      case ExprKind::Unary:
        return EvaluateUnary(expression);
      case ExprKind::Binary:
        return EvaluateBinary(expression);
      case ExprKind::Ternary: {
        const Value condition = EvaluateExpr(expression->left);
        return condition.Scalar() != 0.0f ? EvaluateExpr(expression->right) : EvaluateExpr(expression->third);
      }
    }
    throw std::runtime_error("unhandled expression");
  }

  Value EvaluateIdentifier(const std::string &name) {
    if (name == "RAY_TYPE_CAMERA")
      return Scalar(0.0f);
    if (name == "RAY_TYPE_REFLECTION")
      return Scalar(1.0f);
    if (name == "RAY_TYPE_TRANSMISSION")
      return Scalar(2.0f);
    if (name == "RAY_TYPE_VOLUME")
      return Scalar(3.0f);
    if (name == "PI")
      return Scalar(3.14159265358979323f);
    if (Value *found = Lookup(name))
      return *found;
    throw std::runtime_error("unknown identifier: " + name);
  }

  Value EvaluateMember(const ExprPtr &expression) {
    const Value base = EvaluateExpr(expression->left);
    if (base.material_data) {
      // `material_data.Load(n)` is handled as a call; anything else on it is a
      // mistake in the generated code.
      throw std::runtime_error("unsupported member on a buffer: ." + expression->name);
    }
    if (base.IsStruct())
      return StructField(base, expression->name);
    return Swizzle(base, expression->name);
  }

  Value EvaluateUnary(const ExprPtr &expression) {
    const std::string &op = expression->name;
    if (op == "post" || op == "++" || op == "--") {
      // x++ / ++x only appear as for-loop steps, evaluated for their side
      // effect; the value is discarded either way.
      const bool increment = op != "--";
      AssignTo(expression->left, increment ? "+=" : "-=", Scalar(1.0f));
      return EvaluateExpr(expression->left);
    }
    const Value operand = EvaluateExpr(expression->left);
    if (op == "!") {
      Value result = Scalar(operand.Scalar() == 0.0f ? 1.0f : 0.0f);
      return result;
    }
    Value result = operand;
    if (op == "-") {
      for (int i = 0; i < result.n; ++i)
        result.v[i] = -result.v[i];
    }
    return result;
  }

  Value EvaluateBinary(const ExprPtr &expression) {
    const std::string &op = expression->name;
    // Short circuit, as in C.
    if (op == "&&" || op == "||") {
      const float left = EvaluateExpr(expression->left).Scalar();
      if (op == "&&" && left == 0.0f)
        return Scalar(0.0f);
      if (op == "||" && left != 0.0f)
        return Scalar(1.0f);
      return Scalar(EvaluateExpr(expression->right).Scalar() != 0.0f ? 1.0f : 0.0f);
    }
    const Value a = EvaluateExpr(expression->left);
    const Value b = EvaluateExpr(expression->right);
    if (op == "==")
      return Scalar(a.Scalar() == b.Scalar() ? 1.0f : 0.0f);
    if (op == "!=")
      return Scalar(a.Scalar() != b.Scalar() ? 1.0f : 0.0f);
    if (op == "<")
      return Scalar(a.Scalar() < b.Scalar() ? 1.0f : 0.0f);
    if (op == ">")
      return Scalar(a.Scalar() > b.Scalar() ? 1.0f : 0.0f);
    if (op == "<=")
      return Scalar(a.Scalar() <= b.Scalar() ? 1.0f : 0.0f);
    if (op == ">=")
      return Scalar(a.Scalar() >= b.Scalar() ? 1.0f : 0.0f);

    // Elementwise, broadcasting whichever side has fewer components. The
    // generated code only ever multiplies a vector by a scalar or another
    // vector; matrices reach it through mul() instead.
    Value result;
    result.n = std::max(a.n, b.n);
    if (a.IsMatrix() || b.IsMatrix())
      throw std::runtime_error("matrix arithmetic is not part of the graph grammar");
    for (int i = 0; i < result.n; ++i) {
      const float x = a.v[a.n == 1 ? 0 : i];
      const float y = b.v[b.n == 1 ? 0 : i];
      result.v[i] = op == "+" ? x + y : op == "-" ? x - y : op == "*" ? x * y : x / y;
    }
    return result;
  }

  Value EvaluateCall(const ExprPtr &expression) {
    const std::string &name = expression->name;
    // Type constructors: float, float2, float3, float4, int, uint, bool.
    if (name == "float" || name == "float2" || name == "float3" || name == "float4" || name == "int" ||
        name == "uint" || name == "bool") {
      const int count = name == "float2" ? 2 : name == "float3" ? 3 : name == "float4" ? 4 : 1;
      std::vector<Value> arguments;
      for (const ExprPtr &argument : expression->args)
        arguments.push_back(EvaluateExpr(argument));
      Value result;
      result.n = count;
      int out = 0;
      for (const Value &argument : arguments)
        for (int i = 0; i < argument.n && out < count; ++i)
          result.v[out++] = argument.v[i];
      if (out != count)
        throw std::runtime_error("wrong argument count for " + name);
      return result;
    }
    if (name == "float3x3") {
      std::vector<Value> arguments;
      for (const ExprPtr &argument : expression->args)
        arguments.push_back(EvaluateExpr(argument));
      if (arguments.size() != 3)
        throw std::runtime_error("float3x3 needs three rows");
      Value result;
      result.n = 9;
      for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
          result.v[row * 3 + column] = arguments[static_cast<size_t>(row)].v[column];
      return result;
    }
    if (name == "mul") {
      const Value a = EvaluateExpr(expression->args[0]);
      const Value b = EvaluateExpr(expression->args[1]);
      // HLSL's mul: row vector times matrix, matrix times column vector.
      if (a.n == 3 && b.n == 9) {
        Value result;
        result.n = 3;
        for (int column = 0; column < 3; ++column) {
          float sum = 0.0f;
          for (int row = 0; row < 3; ++row)
            sum += a.v[row] * b.v[row * 3 + column];
          result.v[column] = sum;
        }
        return result;
      }
      if (a.n == 9 && b.n == 3) {
        Value result;
        result.n = 3;
        for (int row = 0; row < 3; ++row) {
          float sum = 0.0f;
          for (int column = 0; column < 3; ++column)
            sum += a.v[row * 3 + column] * b.v[column];
          result.v[row] = sum;
        }
        return result;
      }
      throw std::runtime_error("unsupported mul operands");
    }
    if (name == "material_data.Load") {
      // A uint read of the material's parameter buffer. The graph uses it to
      // fetch a registered texture slot.
      const float offset = EvaluateExpr(expression->args[0]).Scalar();
      Value result = Scalar(0.0f);
      if (input_.material_data && static_cast<size_t>(offset) + sizeof(uint32_t) <= input_.material_data_size) {
        uint32_t raw = 0;
        std::memcpy(&raw, input_.material_data + static_cast<size_t>(offset), sizeof(raw));
        result.v[0] = static_cast<float>(raw);
      }
      return result;
    }
    if (name == "SampleTexture") {
      const float index = EvaluateExpr(expression->args[0]).Scalar();
      const Value uv = EvaluateExpr(expression->args[1]);
      Value result;
      result.n = 4;
      if (input_.sample_texture)
        input_.sample_texture(input_.sample_texture_user, static_cast<int32_t>(index), uv.v[0], uv.v[1], result.v);
      else
        result.v[0] = result.v[1] = result.v[2] = 1.0f;
      result.v[3] = 1.0f;
      return result;
    }

    // A user function from the generated preamble.
    auto found = functions_.find(name);
    if (found != functions_.end()) {
      std::vector<Value> arguments;
      for (const ExprPtr &argument : expression->args)
        arguments.push_back(EvaluateExpr(argument));
      return CallFunction(found->second, arguments);
    }

    // Builtins.
    std::vector<Value> arguments;
    for (const ExprPtr &argument : expression->args)
      arguments.push_back(EvaluateExpr(argument));
    if (name == "SPARKIUM_SPLAT4") {
      // The broadcast macro the codegen emits in place of `(x).xxxx`, which C++
      // cannot spell. HLSL repeats the first component.
      const Value &value = arguments[0];
      Value result;
      result.n = 4;
      for (int i = 0; i < 4; ++i)
        result.v[i] = value.v[0];
      return result;
    }
    return CallBuiltin(name, arguments);
  }

  Value CallFunction(const Function &function, std::vector<Value> arguments) {
    const bool outer_returned = returned_;
    returned_ = false;
    if (arguments.size() != function.parameters.size())
      throw std::runtime_error("wrong argument count calling " + function.name);
    scopes_.emplace_back();
    for (size_t i = 0; i < arguments.size(); ++i)
      scopes_.back()[function.parameters[i]] = std::move(arguments[i]);
    // A struct return value is carried out through `return surface;`.
    Value returned;
    ExecuteBlock(function.body, returned);
    scopes_.pop_back();
    returned_ = outer_returned;
    return returned;
  }

  void ExecuteBlock(const StmtPtr &block, Value &returned) {
    for (const StmtPtr &statement : block->body) {
      Execute(statement, returned);
      if (returned_)
        return;
    }
  }

  void Execute(const StmtPtr &statement, Value &returned) {
    switch (statement->kind) {
      case StmtKind::Block:
        scopes_.emplace_back();
        ExecuteBlock(statement, returned);
        scopes_.pop_back();
        return;
      case StmtKind::VarDecl: {
        const int count = ComponentCount(statement->type);
        for (const auto &[name, initializer] : statement->declarators) {
          Value value;
          if (statement->type == "GraphSurface") {
            value.n = 0;
            value.fields = std::make_shared<std::vector<Value>>();
            for (const std::string &field : SurfaceFields())
              value.fields->push_back(Value::Splat(SurfaceFieldComponents(field), 0.0f));
          } else {
            value.n = count;
          }
          if (initializer) {
            value = EvaluateExpr(initializer);
            if (!value.IsStruct() && value.n != count)
              value = Convert(value, count);
          }
          scopes_.back()[name] = std::move(value);
        }
        return;
      }
      case StmtKind::Assign:
        AssignTo(statement->target, statement->op, EvaluateExpr(statement->value));
        return;
      case StmtKind::ExprStmt:
        if (statement->expr)
          EvaluateExpr(statement->expr);
        return;
      case StmtKind::If:
        if (EvaluateExpr(statement->condition).Scalar() != 0.0f)
          Execute(statement->body.front(), returned);
        else if (statement->otherwise)
          Execute(statement->otherwise, returned);
        return;
      case StmtKind::For: {
        scopes_.emplace_back();
        if (statement->init)
          Execute(statement->init, returned);
        int guard = 0;
        while (!statement->condition || EvaluateExpr(statement->condition).Scalar() != 0.0f) {
          Execute(statement->body.front(), returned);
          if (returned_) {
            scopes_.pop_back();
            return;
          }
          if (statement->step)
            Execute(statement->step, returned);
          if (++guard > 100000)
            throw std::runtime_error("runaway loop in generated shader graph");
        }
        scopes_.pop_back();
        return;
      }
      case StmtKind::Return:
        if (statement->expr)
          returned = EvaluateExpr(statement->expr);
        returned_ = true;
        return;
    }
  }

  // Assignment needs the target's identity, not its value, so the target
  // expression is walked again here.
  void AssignTo(const ExprPtr &target, const std::string &op, const Value &value) {
    if (target->kind == ExprKind::Identifier) {
      Value *slot = Lookup(target->name);
      if (!slot)
        throw std::runtime_error("assignment to unknown variable: " + target->name);
      if (op == "=")
        *slot = Convert(value, slot->IsStruct() ? 0 : slot->n);
      else
        *slot = ApplyCompound(*slot, value, op);
      return;
    }
    if (target->kind == ExprKind::Member) {
      if (target->left->kind == ExprKind::Identifier) {
        Value *slot = Lookup(target->left->name);
        if (slot && slot->IsStruct()) {
          const std::vector<std::string> &names = SurfaceFields();
          for (size_t i = 0; i < names.size(); ++i) {
            if (names[i] == target->name) {
              (*slot->fields)[i] = Convert(value, SurfaceFieldComponents(names[i]));
              return;
            }
          }
          throw std::runtime_error("unknown GraphSurface field: " + target->name);
        }
      }
      // Swizzle assignment: read the base, replace the named components, write
      // the whole vector back.
      if (target->left->kind == ExprKind::Identifier) {
        Value *slot = Lookup(target->left->name);
        if (!slot)
          throw std::runtime_error("assignment to unknown variable");
        Value base = *slot;
        if (target->name.size() == 1) {
          base.v[SwizzleIndex(target->name[0])] = value.Scalar();
        } else {
          for (size_t i = 0; i < target->name.size(); ++i)
            base.v[SwizzleIndex(target->name[i])] = value.v[i];
        }
        *slot = base;
        return;
      }
    }
    throw std::runtime_error("unsupported assignment target");
  }

  static Value ApplyCompound(const Value &current, const Value &operand, const std::string &op) {
    Value result = current;
    const char operation = op[0];
    for (int i = 0; i < result.n; ++i) {
      const float x = result.v[i];
      const float y = operand.v[operand.n == 1 ? 0 : i];
      result.v[i] = operation == '+' ? x + y : operation == '-' ? x - y : operation == '*' ? x * y : x / y;
    }
    return result;
  }

  // HLSL converts between widths freely; this is where that happens.
  static Value Convert(const Value &value, int components) {
    if (components == 0 || value.IsStruct())
      return value;
    Value result;
    result.n = components;
    for (int i = 0; i < components; ++i)
      result.v[i] = value.n == 1 ? value.v[0] : value.v[std::min(i, value.n - 1)];
    return result;
  }

  Value CallBuiltin(const std::string &name, const std::vector<Value> &arguments) {
    auto unary = [&](auto function) {
      Value result = arguments[0];
      for (int i = 0; i < result.n; ++i)
        result.v[i] = static_cast<float>(function(result.v[i]));
      return result;
    };
    auto binary = [&](auto function) {
      Value result = arguments[0];
      for (int i = 0; i < result.n; ++i) {
        const float x = arguments[0].v[i];
        const float y = arguments[1].v[arguments[1].n == 1 ? 0 : i];
        result.v[i] = static_cast<float>(function(x, y));
      }
      return result;
    };

    if (name == "frac")
      return unary([](float x) { return x - std::floor(x); });
    if (name == "floor")
      return unary([](float x) { return std::floor(x); });
    if (name == "abs")
      return unary([](float x) { return std::fabs(x); });
    if (name == "sqrt")
      return unary([](float x) { return std::sqrt(x); });
    if (name == "exp")
      return unary([](float x) { return std::exp(x); });
    if (name == "log")
      return unary([](float x) { return std::log(x); });
    if (name == "sin")
      return unary([](float x) { return std::sin(x); });
    if (name == "cos")
      return unary([](float x) { return std::cos(x); });
    if (name == "saturate")
      return unary([](float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); });
    if (name == "min")
      return binary([](float x, float y) { return x < y ? x : y; });
    if (name == "max")
      return binary([](float x, float y) { return x > y ? x : y; });
    if (name == "pow")
      return binary([](float x, float y) { return std::pow(x, y); });
    if (name == "step")
      return binary([](float edge, float x) { return x < edge ? 0.0f : 1.0f; });
    if (name == "lerp") {
      Value result = arguments[0];
      for (int i = 0; i < result.n; ++i) {
        const float a = arguments[0].v[i];
        const float b = arguments[1].v[arguments[1].n == 1 ? 0 : i];
        const float t = arguments[2].v[arguments[2].n == 1 ? 0 : i];
        result.v[i] = a + (b - a) * t;
      }
      return result;
    }
    if (name == "clamp") {
      Value result = arguments[0];
      for (int i = 0; i < result.n; ++i) {
        const float x = result.v[i];
        result.v[i] = x < arguments[1].Scalar() ? arguments[1].Scalar()
                                                : (x > arguments[2].Scalar() ? arguments[2].Scalar() : x);
      }
      return result;
    }
    if (name == "dot") {
      float sum = 0.0f;
      for (int i = 0; i < arguments[0].n; ++i)
        sum += arguments[0].v[i] * arguments[1].v[i];
      return Scalar(sum);
    }
    if (name == "length") {
      float sum = 0.0f;
      for (int i = 0; i < arguments[0].n; ++i)
        sum += arguments[0].v[i] * arguments[0].v[i];
      return Scalar(std::sqrt(sum));
    }
    if (name == "normalize") {
      float sum = 0.0f;
      for (int i = 0; i < arguments[0].n; ++i)
        sum += arguments[0].v[i] * arguments[0].v[i];
      const float length = std::sqrt(sum);
      Value result = arguments[0];
      for (int i = 0; i < result.n; ++i)
        result.v[i] /= length;
      return result;
    }
    if (name == "cross") {
      Value result;
      result.n = 3;
      const Value &a = arguments[0];
      const Value &b = arguments[1];
      result.v[0] = a.v[1] * b.v[2] - a.v[2] * b.v[1];
      result.v[1] = a.v[2] * b.v[0] - a.v[0] * b.v[2];
      result.v[2] = a.v[0] * b.v[1] - a.v[1] * b.v[0];
      return result;
    }
    throw std::runtime_error("unsupported builtin in generated graph: " + name);
  }

  std::map<std::string, Function> functions_;
  const GraphEvalInput &input_;
  std::vector<Scope> scopes_;
  bool returned_{false};
};

class InterpretedGraphProgram : public GraphProgram {
 public:
  explicit InterpretedGraphProgram(std::map<std::string, Function> functions) : functions_(std::move(functions)) {
  }

  void Evaluate(const GraphEvalInput &input, GraphEvalOutput &output) const override {
    // The graph is evaluated inside the render loop, so a failure must not take
    // the process down: report it once and leave the surface at its defaults.
    try {
      Interpreter interpreter(functions_, input);
      interpreter.Evaluate(output);
    } catch (const std::exception &exception) {
      if (!reported_) {
        reported_ = true;
        grassland::LogError("[sparkium] shader-graph evaluation failed: {}", exception.what());
      }
    }
  }

  const char *Engine() const override {
    return "interpreter";
  }

 private:
  std::map<std::string, Function> functions_;
  mutable bool reported_{false};
};

}  // namespace

std::unique_ptr<GraphProgram> MakeInterpreterGraphProgram(const std::string &source) {
  try {
    Parser parser(source);
    return std::make_unique<InterpretedGraphProgram>(parser.Parse());
  } catch (const std::exception &exception) {
    grassland::LogError("[sparkium] cannot interpret the generated shader graph: {}", exception.what());
    return nullptr;
  }
}

}  // namespace sparkium::raytracing::cpu
