#include <iostream>

#define PRINT_ARG_LIST_DEF(Type)      \
  void PrintArgImpl(const Type &arg); \
  void PrintArgs(const Type &arg);    \
  template <class... Args>            \
  void PrintArgs(const Type &arg, Args &&...args);

#define PRINT_ARG_LIST(Type)                        \
  void PrintArgImpl(const Type &arg) {              \
    std::cout << arg << " ";                        \
  }                                                 \
  void PrintArgs(const Type &arg) {                 \
    PrintArgImpl(arg);                              \
    std::cout << std::endl;                         \
  }                                                 \
  template <class... Args>                          \
  void PrintArgs(const Type &arg, Args &&...args) { \
    PrintArgImpl(arg);                              \
    PrintArgs(args...);                             \
  }

PRINT_ARG_LIST_DEF(int)
PRINT_ARG_LIST_DEF(float)
PRINT_ARG_LIST_DEF(double)
PRINT_ARG_LIST_DEF(std::string)
PRINT_ARG_LIST_DEF(char *)

PRINT_ARG_LIST(int)
PRINT_ARG_LIST(float)
PRINT_ARG_LIST(double)
PRINT_ARG_LIST(std::string)
PRINT_ARG_LIST(char *)

int main() {
  const char str[] = "World!";
  PrintArgs("Hello,", str);
  PrintArgs("123 x 456 =", 123 * 456);
}
