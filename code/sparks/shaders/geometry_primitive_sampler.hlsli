

template <class BufferType>
void SamplePrimitivePower(BufferType direct_lighting_sampler_data, inout float r, out uint primitive_id, out float prob) {
        uint primitive_count = direct_lighting_sampler_data.Load(48);
        BufferReference<BufferType> power_cdf = MakeBufferReference(direct_lighting_sampler_data, 52);
        float total_power = asfloat(power_cdf.Load(primitive_count * 4 - 4));

        uint L = 0, R = primitive_count - 1;
        while (L < R) {
                uint mid = (L + R) / 2;
                float mid_power = asfloat(power_cdf.Load(mid * 4));
                if (r <= mid_power / total_power) {
                R = mid;
                } else {
                L = mid + 1;
                }
        }

        primitive_id = L;
        float high_prob = asfloat(power_cdf.Load(L * 4)) / total_power;
        float low_prob = (L > 0) ? asfloat(power_cdf.Load((L - 1) * 4)) / total_power : 0.0f;
        prob = high_prob - low_prob;

        r = (r - low_prob) / prob;
}

template <class BufferType>
float EvaluatePrimitiveProbability(BufferType direct_lighting_sampler_data, uint primitive_id) {
    uint primitive_count = direct_lighting_sampler_data.Load(48);
    BufferReference<BufferType> power_cdf = MakeBufferReference(direct_lighting_sampler_data, 52);
    float total_power = asfloat(power_cdf.Load(primitive_count * 4 - 4));
    float high_prob = asfloat(power_cdf.Load(primitive_id * 4));
    float low_prob = (primitive_id > 0) ? asfloat(power_cdf.Load((primitive_id - 1) * 4)) : 0.0f;
    return (high_prob - low_prob) / total_power;
}
