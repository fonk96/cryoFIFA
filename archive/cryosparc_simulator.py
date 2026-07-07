    rs = geometry.gen_rs(n_rs)[0]
    Rs = geometry.expmaps(rs)
    geometry.gen_ts(n_ts, e_ts)
    pblob = n.zeros((N_particles, N, N), n.float32)
    pblob_f = n.zeros((N_particles, N, N), n.complex64)

    coords = n.zeros((N**2, 3))
    coords[:,:2] = geometry.gencoords(N,2)
    for p in range(N_particles):
        if p % 100 == 0:
            rc.log("Processed %d / %d particles ..." % (p, N_particles))
        random_pose = n.random.randint(0, rs.shape[0])
        if params['force_single_rot']:
            R_particle = geometry.expmap(n.asarray([0, 0, 0]))
        else:
            R_particle = Rs[random_pose].T
        interp_coords = n.transpose(n.dot(coords, R_particle)) + N//2
        pblob_f[p,:,:] = (spinterp.map_coordinates(vol_f.real,interp_coords[::-1].reshape((3,N**2)),order=1) \
        + 1j*spinterp.map_coordinates(vol_f.imag,interp_coords[::-1].reshape((3,N**2)),order=1)).reshape(N,N)
        particles['alignments3D/pose'][p] = rs[random_pose]
        particles['alignments3D/alpha'][p] = 1.0
        particles['alignments3D/shift'][p] = n.zeros(2)
        if n_volumes == 2:
            particles['alignments3D/localpose'][p] = rs_local[random_local_pose]
            particles['alignments3D/localshift'][p] = n.zeros(2)
            particles['alignments3D/localfulcrum'][p] = fulcrum

    for pidx, particle in enumerate(particles.rows()):
        ctfs = ctf.compute_full_ctf(N, psize*N, particle['ctf/accel_kv'], particle['ctf/cs_mm'], particle['ctf/amp_contrast'], particle['ctf/df1_A'], particle['ctf/df2_A'], particle['ctf/df_angle_rad'], particle['ctf/phase_shift_rad'], particle['ctf/scale'], None).reshape(N,N)
        pblob_f[pidx] *= ctfs

    pblob = fourier.ifft(pblob_f, stack=True)
    noise_var = pblob.std() / snr
    for p in range(pblob.shape[0]):
        for i in range(N):
            pblob[p,i,:] += n.random.normal(0, noise_var, N)
    rc.log("Writing out particles...")
    write_path = os.path.join(job_dir_abs, 'simulated_particles.mrcs')
    write_path_rel = os.path.join(job_dir_rel, 'simulated_particles.mrcs')
    particles['blob/path'] = write_path_rel
    particles['blob/idx'] = list(range(N_particles))
    particles['blob/shape'] = n.asarray([N,N])
    particles['blob/psize_A'] = psize
    particles['blob/sign'] = params.get('blob_sign', -1)
    pblob *= particles['blob/sign'][:, n.newaxis, n.newaxis] # Apply blob/sign to projections before writing out
    mrc.write_mrc(write_path, pblob, psize)