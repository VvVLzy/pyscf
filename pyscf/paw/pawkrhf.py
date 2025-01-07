import time
from copy import deepcopy

import numpy as np

from gpaw import GPAW, PW
from gpaw.mixer import DummyMixer
from gpaw.utilities import unpack_hermitian
from gpaw.hybrids.paw import calculate_paw_stuff
from gpaw.hybrids.scf import apply1

from pyscf.pbc.scf.khf import KRHF

from pyscf import __config__
from pyscf.scf import hf as mol_hf
from pyscf import lib
from pyscf.lib import logger

from ase.units import Ha

# TODO: PySCF kpoint is in 1/Bohr, need to convert to fractional
# coords wrt reciprocal lattice vector before passing into GPAW
# maybe the other way around is better: generate fractional kpts
# in gpaw; then convert and pass into pyscf.


CHECK_COULOMB_IMAG = getattr(__config__, 'pbc_scf_check_coulomb_imag', True)

def project_gpaw_to_AO(calc, cell, kpts, k=0):
    '''
    works for one kpt only for now
    '''
    mesh = calc.wfs.gd.N_c
    nbands = calc.wfs.bd.nbands
    psit_Rn = np.zeros((mesh.prod(), nbands), dtype=np.complex128)
    
    for i in range(nbands):
        # normalize psit_G
        psit_G = calc.wfs.kpt_u[0].psit_nG[i]
        norm = np.linalg.norm(psit_G)
        psit_G /= norm

        # ifft to get real space rep
        psit_R = calc.wfs.pd.ifft(psit_G, q=k)
        normalization = calc.wfs.gd.N_c.prod() / np.sqrt(calc.wfs.gd.volume)
        psit_R *= normalization
        norm = np.sum(psit_R.conj() * psit_R)*calc.wfs.gd.dv # normalized within u.c. indeed
        psit_Rn[:, i] = psit_R.reshape(-1, )

    
    coords = cell.get_uniform_grids(mesh)
    AO_Rj = cell.pbc_eval_gto("GTOval_sph", coords, kpts=kpts)[0]
    c_nj = psit_Rn.T.conj() @ AO_Rj * calc.wfs.gd.dv

    from pyscf.pbc.scf.hf import get_ovlp
    S_inv_ij = np.linalg.inv(get_ovlp(cell, kpts)[k])

    P_nn = c_nj.conj() @ S_inv_ij @ c_nj.T

    return P_nn

def init_gpaw_calc(system, kpts, nbands, e_cut=350, name=None, **kwargs):
    # My entire implementation rely on changing GPAW dtype to comoplex
    # regardless of whether there is only gamma point
    # this is to ensure the GTOs are completely expanded
    mode = PW(e_cut)
    mode.force_complex_dtype = True
    calc = GPAW(mode=mode, kpts=kpts, nbands=nbands,
                txt=name,
                # mixer=Mixer(beta=1, nmaxold=1, weight=0),
                mixer=DummyMixer(),
                **kwargs) # TODO: Make sure that Mixer correctly disables GPAW mixer
    # https://gpaw.readthedocs.io/documentation/densitymix/densitymix.html#densitymix

    calc.atoms = system.copy()

    # set up the objects needed for a calculation: Density, Hamiltonian, WaveFunctions, Setups
    calc.initialize(calc.atoms)

    # Update the positions of the atoms and initialize wave functions, density
    calc.set_positions(calc.atoms)

    return calc

def apply_overlap(wfs, u, calculate_P_ani=True, psit_nG=None):
    """
    ***Adapated from gpaw.overlap.py***
    Apply the overlap operator to a wave function (specified by
    the basis and the expansion coefficient).

    Parameters
    ==========
    wfs: PWWaveFunctions (gpaw/wavefunctions/pw.py)
        Plane-wave wavefunction object
    u: the collective index for spin and kpoint
        wfs.kpt_u[u] gives a KPoint (gpaw/kpoint.py) object that 
        describes wave function for a specific (spin, k) combination
    calculate_P_ani: bool
        When True, the integrals of projector times vectors
        P_ani = <p_ai | psit_nG> for a specific u are calculated.
        When False, existing P_ani are used
    psit_nG: user can provide their own expansion coefficients;
        If None, wfs.kpt_u[u].psit_nG is used.

    """

    '''
    Notes on normalization convention in GPAW:
    https://gitlab.com/gpaw/gpaw/-/blob/master/gpaw/pw/descriptor.py?ref_type=heads#L366
    This is likely to be the right prefactor for "old" GPAW, and complex wave functions.
    For real wave functions, it is more complicated. Also expressed in this function.
    But you could just use pd.integrate
    Also helpful:
    https://gpaw.readthedocs.io/documentation/orthogonalization.html
    '''

    # psi_t at u-th kpoint, u is the combined spin and kpoint index
    kpt = wfs.kpt_u[u]

    # expansion coefficient
    psit_nG = kpt.psit_nG if psit_nG is None else psit_nG

    # b_xG is the resulting expansion coefficient after applying S
    Spsit_nG = np.copy(psit_nG)

    # (taken from GPAW)
    # random initialization of a dictionary with signature
    # {atom: array of len(projectors)}
    shape = psit_nG.shape[0]
    P_ani = wfs.pt.dict(shape)

    if calculate_P_ani:
        # the original function does not update P_ani
        wfs.pt.integrate(psit_nG, P_ani, kpt.q)
    else:
        # for a, P_ni in kpt.P_ani.items():
        #     P_ani[a][:] = P_ni
        # TODO: can probably store the P_ani calculated to prevent
        # redundant calculations afterwards
        pass
        

    for a, P_ni in P_ani.items():
        P_ani[a] = np.dot(P_ni, wfs.setups[a].dO_ii)
    wfs.pt.add(Spsit_nG, P_ani, kpt.q)  # b_xG += sum_ai pt^a_i P_ani

    return Spsit_nG

def apply_pseudo_hamiltonian(wfs, u, ham, psit_nG=None):
    """Apply the pseudo Hamiltonian (without PAW correction) to
    wavefunction (specified by the basis and the expansion coefficient).

    Parameters:

    wfs: PWWaveFunctions (gpaw/wavefunctions/pw.py)
        Plane-wave wavefunction object
    u: the collective index for spin and kpoint
        wfs.kpt_u[u] gives a KPoint (gpaw/kpoint.py) object that 
        describes wave function for a specific (spin, k) combination
    ham: Hamiltonian
    psit_nG: user can provide their own expansion coefficients;
        If None, wfs.kpt_u[u].psit_nG is used.

    """

    # if psit_nG is None:
    #     kpt = wfs.kpt_u[u]
    #     psit_nG = kpt.psit_nG
    # else:
    #     kpt = wfs.kpt_u[u]
    #     psit = kpt.psit.new()
    #     psit.in_memory = False
    #     psit.matrix.array = psit_nG.copy()
    #     kpt.psit = psit

    kpt = wfs.kpt_u[u]
    psit_nG = kpt.psit_nG if psit_nG is None else psit_nG


    Htpsit_nG = np.zeros_like(psit_nG)
    
    # this function can only be used if we use GPAW's wfs
    # maybe we don't want to do that...
    wfs.apply_pseudo_hamiltonian(kpt, ham, psit_nG, Htpsit_nG)

    return Htpsit_nG

def apply_PAW_correction(wfs, u, ham, calculate_P_ani=True, psit_nG=None):
    """
    Apply PAW correction to the wavefunction.

    Parameters
    ==========
    wfs: PWWaveFunctions (gpaw/wavefunctions/pw.py)
        Plane-wave wavefunction object
    u: the collective index for spin and kpoint
        wfs.kpt_u[u] gives a KPoint (gpaw/kpoint.py) object that 
        describes wave function for a specific (spin, k) combination
    ham: Hamiltonian
    calculate_P_ani: bool
        When True, the integrals of projector times vectors
        P_ani = <p_ai | psit_nG> for the specific u are calculated.
        When False, existing P_ani are used
    psit_nG: user can provide their own expansion coefficients;
        If None, wfs.kpt_u[u].psit_nG is used.

    """

    # psi_t at u-th kpoint, u is the combined spin and kpoint index
    kpt = wfs.kpt_u[u]

    # expansion coefficient
    psit_nG = kpt.psit_nG if psit_nG is None else psit_nG

    # b_xG is the resulting expansion coefficient after applying dH
    dHpsit_nG = np.zeros_like(psit_nG)

    # (taken from GPAW)
    # random initialization of a dictionary with signature
    # {atom: array of len(projectors)}
    shape = psit_nG.shape[0]
    P_ani = wfs.pt.dict(shape)

    if calculate_P_ani:
        wfs.pt.integrate(psit_nG, P_ani, kpt.q)
    else:
        for a, P_ni in kpt.P_ani.items():
            P_ani[a][:] = P_ni

    for a, P_ni in P_ani.items():
        dH_ii = unpack_hermitian(ham.dH_asp[a][kpt.s])
        P_ani[a] = np.dot(P_ni, dH_ii)

    wfs.pt.add(dHpsit_nG, P_ani, kpt.q)

    return dHpsit_nG

def update_dens(dens, wfs):
    dens.timer.start('Density')
    with dens.timer('Pseudo density'):
        dens.calculate_pseudo_density(wfs)
    with dens.timer('Atomic density matrices'):
        wfs.calculate_atomic_density_matrices(dens.D_asp)
    with dens.timer('Multipole moments'):
        comp_charge, _Q_aL = dens.calculate_multipole_moments()

    # if isinstance(wfs, LCAOWaveFunctions):
    #     dens.timer.start('Normalize')
    #     dens.normalize(comp_charge)
    #     dens.timer.stop('Normalize')

    # dens.timer.start('Mix')
    dens.mix(comp_charge) # disable GPAW's DM
    # dens.timer.stop('Mix')
    dens.timer.stop('Density')

def apply_to_single_u(ham, wfs, u):
    '''
    Apply the Hamiltonian and Overlap to wavefunction for a 
    single (spin, k) point.
    '''
    Spsit_nG = apply_overlap(wfs, u)

    Htpsit_nG = apply_pseudo_hamiltonian(wfs, u, ham)
    dHpsit_nG = apply_PAW_correction(wfs, u, ham)

    return Htpsit_nG + dHpsit_nG, Spsit_nG


class PAWKRHF(KRHF):
    def __init__(self, cell, calc,
                 kpts=np.zeros((1,3)),
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        super().__init__(cell, kpts, exxdiv)
        self.calc = calc    # GPAW calculator
        self.cell = cell
        
        # construct the basis transformation matrix from GTO to PW
        t1 = time.time()
        self.expand_GTO_in_PW_by_fft()
        t2 = time.time()
        print(f'FFT takes {t2-t1} seconds')

        # normalize AOs such that <AO | S | AO> = 1
        self.normalize_AOs()

        print('Initialized')


    def expand_GTO_in_PW_by_fft(self):
        '''
        Calculate the expansion coefficient of each GTO in
        the auxillary PW basis using FFT; outputs a matrix of size
        (n_PW, n_GTO)
        '''
        # TODO: figure out beyond gamma point (multiple kpts) case.
        
        mesh = self.calc.wfs.gd.N_c
        coords = self.cell.get_uniform_grids(mesh)
        GTOs = self.cell.pbc_eval_gto("GTOval_sph", coords, kpts=self.kpts)

        self.gto2pw = list()
        self.aos_3d = list()    # for reconstructing AO's from pseudo AO's
        self.aos_k = list()     # for verifying PW expansion is complete
        for k in range(len(self.kpts)):
            npw, nao = self.calc.wfs.ng_k[k], self.cell.nao
            gto2pw_k = np.zeros((npw, nao), dtype=np.complex128)
            ao_k = np.zeros((npw, nao), dtype=np.complex128)
            for j in range(nao):
                ao_1d = GTOs[k][:, j]
                ao_3d = ao_1d.reshape(mesh)
                self.aos_3d.append(ao_3d) # need to be more sophisticated for multiple k

                # fft without any normalization in front
                ao_3d_k = self.calc.wfs.pd.fft(ao_3d, q=k)

                # normalized so that <psit | psit> = 1
                # this is a good sanity check that the expansion is complete
                ao_3d_k = ao_3d_k / ao_1d.shape[0]*np.sqrt(self.cell.vol)
                ao_k[:, j] = ao_3d_k

                # # normalized so that <psit | psit> * dv = 1
                # ao_3d_k /= np.sqrt(dv) # normalize AOs will take care of it

                gto2pw_k[:, j] = ao_3d_k
            self.gto2pw.append(gto2pw_k)
            self.aos_k.append(ao_k)

    def normalize_AOs(self):
        '''
        Rescale each AOs so that they are normalized,
        But we don't orthogonalize different AOs.
        In practice, this means that the diagonal element
        of the overlap matrix should all be 1's, for each kpt
        '''
        S = self.get_ovlp()

        for k in range(len(S)):
            assert(S[k].shape[0] == S[k].shape[1])
            for ao in range(S[k].shape[0]):
                norm = S[k][ao, ao]
                self.gto2pw[k][:, ao] /= np.sqrt(norm)
                # self.calc.kpt_u[k].psit

        S_new = self.get_ovlp()
        for k in range(len(S_new)):
            for ao in range(S_new[k].shape[0]):
                assert(np.isclose(S_new[k][ao, ao], 1))

    def get_h_matrix(self):
        '''
        This should return (kpts, nao, nao) size array
        '''
        nkpts = len(self.kpts)
        nao = self.cell.nao
        calc = self.calc
        h = np.zeros((nkpts, nao, nao), dtype=np.complex128)

        for k in range(nkpts):
            dv = calc.wfs.kpt_u[k].psit.dv
            # recall each column of gto2pw[k] is PW coefficient of AO, gto2pw[k].shape = (N_PW, nao)
            psit_nG = self.gto2pw[k].T.copy()
            Htpsit_nG = apply_pseudo_hamiltonian(calc.wfs, k, calc.hamiltonian, psit_nG=psit_nG).T
            dHpsit_nG = apply_PAW_correction(calc.wfs, k, calc.hamiltonian, psit_nG=psit_nG, calculate_P_ani=True).T
            Hpsit_nG = Htpsit_nG + dHpsit_nG    # H |psit_nG>.shape = (N_PW, nao)
            
            h[k, :, :] = self.gto2pw[k].conj().T @ Hpsit_nG * dv

        return h
    
    def update_calc(self, mo_coeff_kpts, mo_occ_kpts, mo_energy_kpts, fermi):
        assert(mo_coeff_kpts is not None)
        assert(mo_occ_kpts is not None)
        assert(len(mo_coeff_kpts) == len(mo_occ_kpts))

        # self.mo_coeff is (kpt, nao, nmo) list of 2D array
        # self.mo_occ is (kpt, nao) list of 1D array

        wfs = self.calc.wfs
        dens = self.calc.density
        ham = self.calc.hamiltonian

        for k in range(len(mo_coeff_kpts)):
            mo_coeff_pw = self.gto2pw[k] @ mo_coeff_kpts[k]
            kpt = wfs.kpt_u[k]

            # update |psit>
            # Caution: messing with memory big time
            # psit = kpt.psit.new()
            # psit.in_memory = False
            # psit.matrix.array = mo_coeff_pw.T.copy()
            # kpt.psit = psit
            psit = kpt.psit.new(buf=mo_coeff_pw.T)
            kpt.psit[:] = psit
            
            # update < p | psit >
            # TODO: potential speed up by using stored <p | AO>
            P = kpt.projections.new()
            kpt.psit.matrix_elements(wfs.pt, out=P)
            kpt.projections = P
            
            # update occupation number and band energy
            kpt.f_n = mo_occ_kpts[k].copy()
            kpt.eps_n = mo_energy_kpts[k].copy()

        wfs.fermi_levels = np.array([fermi])

        # update dens and ham
        update_dens(dens, wfs)
        ham.update(dens)

    def get_occ(self, mo_energy_kpts=None, mo_coeff_kpts=None):
        '''Label the occupancies for each orbital for sampled k-points.

        This is a k-point version of scf.hf.SCF.get_occ
        '''
        if mo_energy_kpts is None: mo_energy_kpts = self.mo_energy

        nkpts = len(mo_energy_kpts)
        nocc = self.cell.tot_electrons(nkpts) // 2

        mo_energy = np.sort(np.hstack(mo_energy_kpts))
        fermi = mo_energy[nocc-1]
        mo_occ_kpts = []
        for mo_e in mo_energy_kpts:
            mo_occ_kpts.append((mo_e <= fermi).astype(np.double) * 2)

        if nocc < mo_energy.size:
            logger.info(self, 'HOMO = %.12g  LUMO = %.12g',
                        mo_energy[nocc-1], mo_energy[nocc])
            if mo_energy[nocc-1]+1e-3 > mo_energy[nocc]:
                logger.warn(self, 'HOMO %.12g == LUMO %.12g',
                            mo_energy[nocc-1], mo_energy[nocc])
        else:
            logger.info(self, 'HOMO = %.12g', mo_energy[nocc-1])

        if self.verbose >= logger.DEBUG:
            np.set_printoptions(threshold=len(mo_energy))
            logger.debug(self, '     k-point                  mo_energy')
            for k,kpt in enumerate(self.cell.get_scaled_kpts(self.kpts)):
                logger.debug(self, '  %2d (%6.3f %6.3f %6.3f)   %s %s',
                            k, kpt[0], kpt[1], kpt[2],
                            np.sort(mo_energy_kpts[k][mo_occ_kpts[k]> 0]),
                            np.sort(mo_energy_kpts[k][mo_occ_kpts[k]==0]))
            np.set_printoptions(threshold=1000)

        #######################
        # Append script to communicate with the GPAW calculator
        #######################
        self.update_calc(mo_coeff_kpts, mo_occ_kpts, mo_energy_kpts, fermi)

        return mo_occ_kpts

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
                diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
                fock_last=None):
        
        h1e_kpts, s_kpts, vhf_kpts, dm_kpts = h1e, s1e, vhf, dm
        # Parts to modify
        # if h1e_kpts is None: h1e_kpts = mf.get_hcore()
        # if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)
        # f_kpts = h1e_kpts + vhf_kpts
        #############
        f_kpts = self.get_h_matrix() 
        #################

        if cycle < 0 and diis is None:  # Not inside the SCF iteration
            return f_kpts

        if diis_start_cycle is None:
            diis_start_cycle = self.diis_start_cycle
        if level_shift_factor is None:
            level_shift_factor = self.level_shift
        if damp_factor is None:
            damp_factor = self.damp
        if s_kpts is None: s_kpts = self.get_ovlp()
        if dm_kpts is None: dm_kpts = self.make_rdm1()

        if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4 and fock_last is not None:
            f_kpts = [mol_hf.damping(f, f_prev, damp_factor) for f,f_prev in zip(f_kpts,fock_last)]
        if diis and cycle >= diis_start_cycle:
            f_kpts = diis.update(s_kpts, dm_kpts, f_kpts, self, h1e_kpts, vhf_kpts, f_prev=fock_last)
        if abs(level_shift_factor) > 1e-4:
            f_kpts = [mol_hf.level_shift(s, dm_kpts[k], f_kpts[k], level_shift_factor)
                    for k, s in enumerate(s_kpts)]
        return lib.asarray(f_kpts)

    def get_ovlp(self, cell=None, kpts=None):
        '''Get the overlap AO matrices at sampled k-points.

        Args:
            kpts : (nkpts, 3) ndarray

        Returns:
            ovlp_kpts : (nkpts, nao, nao) ndarray
        '''
        calc = self.calc
        nao = self.cell.nao
        nkpts = len(self.kpts)
        s = np.zeros((nkpts, nao, nao), dtype=np.complex128)
        
        for k in range(nkpts):
            # GPAW normalization
            dv = calc.wfs.kpt_u[k].psit.dv

            # the copy is somehow necessary to avoid bug when doing integral in gpaw
            psit_nG = self.gto2pw[k].T.copy()
            Spsit_nG = apply_overlap(calc.wfs, k, psit_nG=psit_nG).T
            s[k, :, :] = self.gto2pw[k].conj().T @ Spsit_nG * dv

        return s
    
    def energy_elec(self, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
        '''
        GPAW energy
        '''
        # if dm_kpts is None: dm_kpts = mf.make_rdm1()
        # if h1e_kpts is None: h1e_kpts = mf.get_hcore()
        # if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

        # nkpts = len(dm_kpts)
        # e1 = 1./nkpts * np.einsum('kij,kji', dm_kpts, h1e_kpts)
        # e_coul = 1./nkpts * np.einsum('kij,kji', dm_kpts, vhf_kpts) * 0.5
        # mf.scf_summary['e1'] = e1.real
        # mf.scf_summary['e2'] = e_coul.real
        # logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
        # if CHECK_COULOMB_IMAG and abs(e_coul.imag > mf.cell.precision*10):
        #     logger.warn(mf, "Coulomb energy has imaginary part %s. "
        #                 "Coulomb integrals (e-e, e-N) may not converge !",
        #                 e_coul.imag)
        # return (e1+e_coul).real, e_coul.real
        

        calc = self.calc
        wfs = calc.wfs
        kpt = wfs.kpt_u[0] # works for one kpt only!!
        xc = calc.hamiltonian.xc
        if wfs.kpt_u[0].eps_n is None:
            # initialization
            s1e = self.get_ovlp()
            fock = self.get_fock()
            mo_energy, mo_coeff = self.eig(fock, s1e)
            mo_occ = self.get_occ(mo_energy, mo_coeff) # to update calc.wfs, dens, ham
        # wfs.eigensolver.iterate(ham, wfs)

        # EXX/Hybrid energy correction
        if xc.type == 'HYB' and (kpt.s, kpt.k) not in xc.v_sknG:
            assert not any(s == kpt.s for s, k in xc.v_sknG)
            paw_s = calculate_paw_stuff(wfs, xc.dens)
            evc, evv, ekin, _ = apply1(
                kpt, None,
                wfs,
                xc.coulomb, xc.sym,
                paw_s[kpt.s])
            if kpt.s == 0:
                xc.evc = 0.0
                xc.evv = 0.0
                xc.ekin = 0.0
            scale = 2 / wfs.nspins * xc.exx_fraction
            xc.evc += evc * scale
            xc.evv += evv * scale
            xc.ekin += ekin * scale
        #     xc.v_sknG = {(kpt.s, k): v_nG
        #                     for k, v_nG in v_knG.items()}
        # v_nG = xc.v_sknG.pop((kpt.s, kpt.k))

        # e_entropy = wfs.calculate_occupation_numbers(fix_fermi_level=False) ### !!
        e_entropy = 0 # TODO: True for ground state calculation
        calc.hamiltonian.get_energy(e_entropy, wfs,
                                    kin_en_using_band=True,
                                    e_sic=0.0) # not sure what e_sic is yet
        e_total = calc.hamiltonian.e_total_extrapolated
        e_coul = calc.hamiltonian.e_coulomb

        '''
        these energy are relative energies: 
        "The GPAW code calculates energies relative to the energy of separated reference atoms,
        where each atom is in a spin-paired, neutral, and spherically symmetric state - the state
        that was used to generate the setup"

        we could try to recover the absolute (true) total energy by accounting
        for the reference eneregy contained in the PAW dataset.
        but even then, the energy should not be the same as the Hartree Fock total energy:
        The true number is always lower, because most atoms have a spin-polarized and non-spherical
        symmetric ground state, with an energy that is lower than that of the spin-paired,
        and spherically symmetric reference atom.
        '''
        # TODO: these energies already include nuc-nuc, need to modify hf.e_total
        return e_total.real, e_coul.real
    
    def get_veff(self, cell=None, dm_kpts=None, dm_last=0, vhf_last=0, hermi=1,
                 kpts=None, kpts_band=None):
        '''Hartree-Fock potential matrix for the given density matrix.
        See :func:`scf.hf.get_veff` and :func:`scf.hf.RHF.get_veff`
        '''
        # if dm_kpts is None:
        #     dm_kpts = self.make_rdm1()
        # vj, vk = self.get_jk(cell, dm_kpts, hermi, kpts, kpts_band)
        # return vj - vk * .5
        # TODO: "trivialize" this function for now...
        return 0
    
    def get_hcore(self, cell=None, kpts=None):
        '''Get the core Hamiltonian AO matrices at sampled k-points.

        Args:
            kpts : (nkpts, 3) ndarray

        Returns:
            hcore : (nkpts, nao, nao) ndarray
        '''
        # if cell is None: cell = mf.cell
        # if kpts is None: kpts = mf.kpts
        # if cell.pseudo:
        #     nuc = lib.asarray(mf.with_df.get_pp(kpts))
        # else:
        #     nuc = lib.asarray(mf.with_df.get_nuc(kpts))
        # if len(cell._ecpbas) > 0:
        #     from pyscf.pbc.gto import ecp
        #     nuc += lib.asarray(ecp.ecp_int(cell, kpts))
        # t = lib.asarray(cell.pbc_intor('int1e_kin', 1, 1, kpts))
        # return nuc + t
        return 0
    
    def energy_tot(self, dm=None, h1e=None, vhf=None):
        r'''Total Hartree-Fock energy, electronic part plus nuclear repulsion
        See :func:`scf.hf.energy_elec` for the electron part

        Note this function has side effects which cause self.scf_summary updated.

        '''
        # nuc = self.energy_nuc()
        nuc = 0 # GPAW energy already includes nuc-nuc interaction
        self.scf_summary['nuc'] = nuc.real

        e_tot = self.energy_elec(dm, h1e, vhf)[0] + nuc + self.calc.get_reference_energy()/Ha
        if self.do_disp():
            if 'dispersion' in self.scf_summary:
                e_tot += self.scf_summary['dispersion']
            else:
                e_disp = self.get_dispersion()
                self.scf_summary['dispersion'] = e_disp
                e_tot += e_disp

        return e_tot