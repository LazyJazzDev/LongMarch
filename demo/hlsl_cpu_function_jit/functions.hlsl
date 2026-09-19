// export and __extern_cpp are Slang extensions to HLSL:
// retain this function and export its unmangled name.
export __extern_cpp float evaluate(float x) {
  return x * x + 1.0f;
}

export __extern_cpp float multiplyAdd(float x, float scale, float bias) {
  return x * scale + bias;
}
