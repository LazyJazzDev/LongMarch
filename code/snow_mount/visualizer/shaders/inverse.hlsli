#ifndef INVERSE_HLSLI
#define INVERSE_HLSLI

// Returns the inverse of a matrix, by using the algorithm of calculating the classical
// adjoint and dividing by the determinant. The contents of the matrix are changed.
float2x2 inverse(float2x2 m) {
  float2x2 adj;  // The adjoint matrix (inverse after dividing by determinant)

  // Create the transpose of the cofactors, as the classical adjoint of the matrix.
  adj[0][0] = m[1][1];
  adj[0][1] = -m[0][1];

  adj[1][0] = -m[1][0];
  adj[1][1] = m[0][0];

  // Calculate the determinant as a combination of the cofactors of the first row.
  float det = (adj[0][0] * m[0][0]) + (adj[0][1] * m[1][0]);

  // Divide the classical adjoint matrix by the determinant.
  // If determinant is zero, matrix is not invertable, so leave it unchanged.
  return (det != 0.0f) ? (adj * (1.0f / det)) : m;
}

// Returns the determinant of a 2x2 matrix.
float spvDet2x2(float a1, float a2, float b1, float b2) {
  return a1 * b2 - b1 * a2;
}

// Returns the inverse of a matrix, by using the algorithm of calculating the classical
// adjoint and dividing by the determinant. The contents of the matrix are changed.
float3x3 inverse(float3x3 m) {
  float3x3 adj;  // The adjoint matrix (inverse after dividing by determinant)

  // Create the transpose of the cofactors, as the classical adjoint of the matrix.
  adj[0][0] = spvDet2x2(m[1][1], m[1][2], m[2][1], m[2][2]);
  adj[0][1] = -spvDet2x2(m[0][1], m[0][2], m[2][1], m[2][2]);
  adj[0][2] = spvDet2x2(m[0][1], m[0][2], m[1][1], m[1][2]);

  adj[1][0] = -spvDet2x2(m[1][0], m[1][2], m[2][0], m[2][2]);
  adj[1][1] = spvDet2x2(m[0][0], m[0][2], m[2][0], m[2][2]);
  adj[1][2] = -spvDet2x2(m[0][0], m[0][2], m[1][0], m[1][2]);

  adj[2][0] = spvDet2x2(m[1][0], m[1][1], m[2][0], m[2][1]);
  adj[2][1] = -spvDet2x2(m[0][0], m[0][1], m[2][0], m[2][1]);
  adj[2][2] = spvDet2x2(m[0][0], m[0][1], m[1][0], m[1][1]);

  // Calculate the determinant as a combination of the cofactors of the first row.
  float det = (adj[0][0] * m[0][0]) + (adj[0][1] * m[1][0]) + (adj[0][2] * m[2][0]);

  // Divide the classical adjoint matrix by the determinant.
  // If determinant is zero, matrix is not invertable, so leave it unchanged.
  return (det != 0.0f) ? (adj * (1.0f / det)) : m;
}

// Returns the determinant of a 3x3 matrix.
float spvDet3x3(float a1, float a2, float a3, float b1, float b2, float b3, float c1, float c2, float c3) {
  return a1 * spvDet2x2(b2, b3, c2, c3) - b1 * spvDet2x2(a2, a3, c2, c3) + c1 * spvDet2x2(a2, a3, b2, b3);
}

// Returns the inverse of a matrix, by using the algorithm of calculating the classical
// adjoint and dividing by the determinant. The contents of the matrix are changed.
float4x4 inverse(float4x4 m) {
  float4x4 adj;  // The adjoint matrix (inverse after dividing by determinant)

  // Create the transpose of the cofactors, as the classical adjoint of the matrix.
  adj[0][0] = spvDet3x3(m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3]);
  adj[0][1] = -spvDet3x3(m[0][1], m[0][2], m[0][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3]);
  adj[0][2] = spvDet3x3(m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[3][1], m[3][2], m[3][3]);
  adj[0][3] = -spvDet3x3(m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3]);

  adj[1][0] = -spvDet3x3(m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], m[3][3]);
  adj[1][1] = spvDet3x3(m[0][0], m[0][2], m[0][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], m[3][3]);
  adj[1][2] = -spvDet3x3(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[3][0], m[3][2], m[3][3]);
  adj[1][3] = spvDet3x3(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3]);

  adj[2][0] = spvDet3x3(m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], m[3][3]);
  adj[2][1] = -spvDet3x3(m[0][0], m[0][1], m[0][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], m[3][3]);
  adj[2][2] = spvDet3x3(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[3][0], m[3][1], m[3][3]);
  adj[2][3] = -spvDet3x3(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3]);

  adj[3][0] = -spvDet3x3(m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2]);
  adj[3][1] = spvDet3x3(m[0][0], m[0][1], m[0][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2]);
  adj[3][2] = -spvDet3x3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[3][0], m[3][1], m[3][2]);
  adj[3][3] = spvDet3x3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);

  // Calculate the determinant as a combination of the cofactors of the first row.
  float det = (adj[0][0] * m[0][0]) + (adj[0][1] * m[1][0]) + (adj[0][2] * m[2][0]) + (adj[0][3] * m[3][0]);

  // Divide the classical adjoint matrix by the determinant.
  // If determinant is zero, matrix is not invertable, so leave it unchanged.
  return (det != 0.0f) ? (adj * (1.0f / det)) : m;
}

#endif
