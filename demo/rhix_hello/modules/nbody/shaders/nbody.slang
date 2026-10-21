struct GlobalSettings {
  int num_particle;
  float delta_t;
  float gravity;
};

RHI_BYTE_ADDRESS_BUFFER(0, 0, positions);
RHI_RW_BYTE_ADDRESS_BUFFER(0, 1, velocities);
RHI_RW_BYTE_ADDRESS_BUFFER(0, 2, positions_new);
RHI_UNIFORM(0, 3, GlobalSettings, g_param);

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

  velocities.Store<float3>(DTid.x * sizeof(float3), vel);
  positions_new.Store<float3>(DTid.x * sizeof(float3), pos);
}
