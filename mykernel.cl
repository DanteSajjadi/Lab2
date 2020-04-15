/* widthA=heightB for valid matrix multiplication */
__kernel void simpleMultiply(
    __global float *c,
    __local float* local_result)
{
    __local float lsum[4];
    float sum = 0.0f;
    int lsize,gsize,ngroups;
    int gid = get_global_id (0);
    int lid  = get_local_id(0);
    // number of workers in each work group (4)
    lsize = get_local_size(0);
    // total number of workers (1024)
    gsize = get_global_size(0); 
    // number of groups (256)
    ngroups = get_num_groups(0);
    if(gid == 0)
        for( int i = 0; i < ngroups ; i++)
                 local_result[i] = 0;

    if(gid%2 == 0)
    {
          sum += 1.0f / (1.0f + 2.0f*gid);
          lsum[lid] = sum;
    }
    else
    {
          sum -= 1.0f / (1.0f + 2.0f*gid);
            lsum[lid] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    //printf("I am worker number %d locally %d and my calculation is %f\n", gid, lid, sum);
    if(lid == 0)
    {
        sum = 0.0f;
        for(int i = 0; i<lsize ; i++)
                sum += lsum[i];
        local_result[gid/4] = sum;
         printf("My work group sum is %f \n", local_result[gid/4]);
    }
    /* Make sure local processing has completed */
    barrier(CLK_GLOBAL_MEM_FENCE);
    if(gid == 1023)
    {
        sum = 0.0f;
        for(int i = 0; i<ngroups; i++)
              sum += local_result[i];
        *c = sum*4;
    }
                
}