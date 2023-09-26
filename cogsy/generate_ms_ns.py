import cogsworth

j = 1

for i in range(10):
    p = cogsworth.pop.Population(1000, processes=6, m1_cutoff=4, final_kstar1=[13], final_kstar2=[1])
    p.sample_initial_binaries()
    p.perform_stellar_evolution()

    mask = (p.final_bpp["kstar_1"] == 13) & (p.final_bpp["kstar_2"] == 1) & ~p.disrupted
    print(f"{mask.sum()}/{len(mask)}")

    alt_mask = (p.bpp["kstar_1"] == 13) & (p.bpp["kstar_2"] == 1) & (p.bpp["sep"] > 0)
    print(f"\t{alt_mask.sum()}/{len(alt_mask)}")


    if mask.sum() > 0:
        q = p[p.bin_nums[mask]]
        print(q.bpp)
        q.bpp.to_hdf(f"options_{j}.h5", key="df")
        j += 1