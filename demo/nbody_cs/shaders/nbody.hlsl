struct GlobalSettings {
  int num_particle;
  float delta_t;
  float gravity;
};

ByteAddressBuffer positions : register(t0, space0);
RWByteAddressBuffer velocities : register(u0, space1);
RWByteAddressBuffer positions_new : register(u0, space2);
ConstantBuffer<GlobalSettings> g_param : register(b0, space3);

#define blocksize 128

groupshared float3 shared_pos[blocksize];

[numthreads(blocksize, 1, 1)] void CSMain(uint3 Gid
                                          : SV_GroupID, uint3 DTid
                                          : SV_DispatchThreadID, uint3 GTid
                                          : SV_GroupThreadID, uint GI
                                          : SV_GroupIndex) {
  float3 pos = positions.Load<float3>(DTid.x * sizeof(float3));
  float3 vel = velocities.Load<float3>(DTid.x * sizeof(float3));

  float3 accel = float3(0.0, 0.0, 0.0);

  for (int i = 0; i < g_param.num_particle / blocksize; i++) {
    shared_pos[GI] = positions.Load<float3>((i * blocksize + GI) * sizeof(float3));
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (int j = 0; j < blocksize; j++) {
      float3 pos_j = shared_pos[j];
      float3 diff = pos - pos_j;
      float l = rsqrt(dot(diff, diff) + 0.00125f * 0.00125f);
      l = l * l * l * (-g_param.delta_t * g_param.gravity);
      accel += diff * l;
    }
    GroupMemoryBarrierWithGroupSync();
  }

  vel += accel;
  pos += vel * g_param.delta_t;

  velocities.Store<float3>(DTid.x * sizeof(float3), vel);  //*/
  positions_new.Store<float3>(DTid.x * sizeof(float3), pos);
}
