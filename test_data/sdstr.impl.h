namespace sdstr {
  void _waitall(MPI_Request *reqs) {
    MPI_Status statuses[128]; /* big number */
    m::Waitall(26, reqs, statuses) ;
  }

  void ini(MPI_Comm cart)  {
    packsizes = new PinnedHostBuffer4<int>(27);

    mpDeviceMalloc(&remote_particles);
    CC(cudaMalloc(&compressed_cellcounts, sizeof(compressed_cellcounts[0]) * XS*YS*ZS));


    for(int i = 0; i < 27; ++i) {
      int d[3] = { (i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1 };
      recv_tags[i] = (3 - d[0]) % 3 + 3 * ((3 - d[1]) % 3 + 3 * ((3 - d[2]) % 3));
      int co_ne[3], ranks[3] = {rankx, ranky, rankz};
      for(int c = 0; c < 3; ++c) co_ne[c] = ranks[c] + d[c];
      m::Cart_rank(cart, co_ne, ra_ne + i) ;

      int nhalodir[3] =  {
	d[0] != 0 ? 1 : XS,
	d[1] != 0 ? 1 : YS,
	d[2] != 0 ? 1 : ZS
      };

      int nhalocells = nhalodir[0] * nhalodir[1] * nhalodir[2];
      int safety_factor = 2;
      int estimate = numberdensity * safety_factor * nhalocells;
      CC(cudaMalloc(&packbuffers[i].scattered_indices, sizeof(int) * estimate));

      if (i && estimate) {
	CC(cudaHostAlloc(&ssnd[i], sizeof(float) * 6 * estimate, cudaHostAllocMapped));
	CC(cudaHostGetDevicePointer(&packbuffers[i].buffer, ssnd[i], 0));
	CC(cudaHostAlloc(&rrcv[i], sizeof(float) * 6 * estimate, cudaHostAllocMapped));
	CC(cudaHostGetDevicePointer(&unpackbuffers[i].buffer, rrcv[i], 0));
      } else {
	CC(cudaMalloc(&packbuffers[i].buffer, sizeof(float) * 6 * estimate));
	unpackbuffers[i].buffer = packbuffers[i].buffer;
	ssnd[i] = NULL;
	rrcv[i] = NULL;
      }
    }

    setup_texture(k_sdstr::texPP1,  float);
    setup_texture(k_sdstr::texPP2, float2);

    CC(cudaEventCreate(&evpacking, cudaEventDisableTiming));
    CC(cudaEventCreate(&evsizes, cudaEventDisableTiming));
  }

  void _post_recv(MPI_Comm cart) {
    for(int i = 1, c = 0; i < 27; ++i)
	m::Irecv(rrcv_sz + i, 1, MPI_INTEGER, ra_ne[i],
		 950 + recv_tags[i], cart, rrcv_req + c++);

    for(int i = 1, c = 0; i < 27; ++i)
	m::Irecv(rrcv[i], MAX_PART_NUM, MPI_FLOAT, ra_ne[i],
		 950 + recv_tags[i] + 333, cart, recvmsgreq + c++);

  }

  void pack(Particle *pp, int n, MPI_Comm cart) {
    size_t offset;
    if (n) {
      CC(cudaBindTexture(&offset, &k_sdstr::texPP1, pp,
			 &k_sdstr::texPP1.channelDesc,
			 sizeof(float) * 6 * n));
      CC(cudaBindTexture(&offset, &k_sdstr::texPP2, pp,
			 &k_sdstr::texPP2.channelDesc,
			 sizeof(float) * 6 * n));
    }

    CC(cudaMemcpyToSymbolAsync(k_sdstr::pack_buffers, packbuffers,
			       sizeof(PackBuffer) * 27, 0, H2D));
    k_sdstr::setup<<<1, 32>>>();
    if (n) k_sdstr::scatter_halo_indices_pack<<<k_cnf(n)>>>(n);
    k_sdstr::tiny_scan<<<1, 32>>>(n, packsizes->DP);
    CC(cudaEventRecord(evsizes));
    if (n) k_sdstr::pack<<<k_cnf(3 * n)>>>(n, n * 3);

    CC(cudaEventRecord(evpacking));
    CC(cudaEventSynchronize(evsizes));
  }

  void send(MPI_Comm cart, int *nbulk, bool firstcall) {
    if (!firstcall) _waitall(ssnd_req);
    for(int i = 0; i < 27; ++i) ssnd_sz[i] = packsizes->D[i];
    *nbulk = rrcv_sz[0] = ssnd_sz[0];
    for(int i = 1, cnt = 0; i < 27; ++i)
      m::Isend(ssnd_sz + i, 1, MPI_INTEGER, ra_ne[i],
	       950 + i, cart, &ssnd_req[cnt++]);
    CC(cudaEventSynchronize(evpacking));
    if (!firstcall) _waitall(sendmsgreq);
    for(int i = 1, cnt = 0; i < 27; ++i)
      m::Isend(ssnd[i], ssnd_sz[i] * 6, MPI_FLOAT, ra_ne[i],
	       950 + i + 333, cart, &sendmsgreq[cnt++]);
  }

  void bulk(Particle *pp, int n, int *cellcounts, uchar4 *subindices) {
    CC(cudaMemsetAsync(cellcounts, 0, sizeof(int) * XS * YS * ZS));
    if (n)
      k_common::subindex_local<false><<<k_cnf(n)>>>
	(n, (float2*)pp, cellcounts, subindices);
  }

  int recv_count(int nbulk, int *nhalo_padded, int *nhalo, uchar4 *subindices) {
    int nexpected;
    _waitall(rrcv_req);
    {
      static int usize[27], ustart[28], ustart_padded[28];

      usize[0] = 0;
      for(int i = 1; i < 27; ++i)
	usize[i] = rrcv_sz[i];

      ustart[0] = 0;
      for(int i = 1; i < 28; ++i)
	ustart[i] = ustart[i - 1] + usize[i - 1];

      nexpected = nbulk + ustart[27];
      *nhalo = ustart[27];

      ustart_padded[0] = 0;
      for(int i = 1; i < 28; ++i)
	ustart_padded[i] = ustart_padded[i - 1] + 32 * ((usize[i - 1] + 31) / 32);

      *nhalo_padded = ustart_padded[27];

      CC(cudaMemcpyToSymbolAsync(k_sdstr::unpack_start, ustart,
				 sizeof(int) * 28, 0, H2D));

      CC(cudaMemcpyToSymbolAsync(k_sdstr::unpack_start_padded, ustart_padded,
				 sizeof(int) * 28, 0, H2D));
    }

    return nexpected;
  }

  void recv_unpack(Particle *pp, float4 *zip0, ushort4 *zip1, int n,
		   int *cellstarts, int *cellcounts,
		   int nold, int nhalo_padded, int nhalo,
		   uchar4   *subindices, uchar4 *subindices_remote,
		   uint *scattered_indices,
		   MPI_Comm cart) {
    _waitall(recvmsgreq);
    CC(cudaMemcpyToSymbolAsync(k_sdstr::unpack_buffers, unpackbuffers,
			       sizeof(UnpackBuffer) * 27, 0, H2D));
    if (nhalo)
      k_sdstr::subindex_remote<<<k_cnf(nhalo_padded)>>>
	(nhalo_padded, nhalo, cellcounts, (float2*)remote_particles, subindices_remote);

    k_common::compress_counts<<<k_cnf(XS*YS*ZS)>>>
      (XS*YS*ZS, (int4*)cellcounts, (uchar4*)compressed_cellcounts);

    k_scan::scan(compressed_cellcounts, XS*YS*ZS, (uint*)cellstarts);

    if (nold)
      k_sdstr::scatter_indices<<<k_cnf(nold)>>>
	(false, subindices, nold, cellstarts, scattered_indices);

    if (nhalo)
      k_sdstr::scatter_indices<<<k_cnf(nhalo)>>>
	(true, subindices_remote, nhalo, cellstarts, scattered_indices);

    if (n)
      k_sdstr::gather_particles<<<k_cnf(n)>>>(scattered_indices,
					      (float2*)remote_particles,
					      n,
					      (float2 *)pp,
					      zip0,
					      zip1);
    _post_recv(cart);
  }

  void _cancel_recv() {
    _waitall(ssnd_req);
    _waitall(sendmsgreq);
    for(int i = 0; i < 26; ++i) m::Cancel(rrcv_req + i) ;
    for(int i = 0; i < 26; ++i) m::Cancel(recvmsgreq + i) ;
  }


  void fin(bool firstcall) {
    CC(cudaEventDestroy(evpacking));
    CC(cudaEventDestroy(evsizes));
    if (!firstcall) _cancel_recv();

    for(int i = 0; i < 27; ++i) {
      CC(cudaFree(packbuffers[i].scattered_indices));
      if (i) CC(cudaFreeHost(packbuffers[i].buffer));
      else   CC(cudaFree(packbuffers[i].buffer));
    }

    delete packsizes;
    CC(cudaFree(compressed_cellcounts));
    CC(cudaFree(remote_particles));
  }
}
