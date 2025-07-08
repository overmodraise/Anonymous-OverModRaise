#include "HEAAN/bootstrap.h"
#include "setup/rns.h"

#include <iostream>
#include <chrono>
#include <omp.h>


template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S>
void conventional_bootstrap_test()
{
    std::cout << "Measuring error on conventional bootstrap" << std::endl;
    std::cout << "LOGN : " << LOGN << std::endl;
    std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
    std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

    const int N = 1 << LOGN;

    int s[N], s_sq[N];
    HEAAN<LOGQ,N>::keygen(H,s);

    Message<LOGN> z, z_out;
    set_test_message(z);

    R_Q<LOGq, N> pt;
    R_Q_square<LOGq,N> ct;
    encode(z,Delta,pt);
    HEAAN<LOGq,N>::enc(pt,s,ct);

    // ModRaise
    R_Q_square<LOGQ,N> ct_modraise;
    mod_raise<LOGq,LOGQ,N>(ct,ct_modraise);

    std::cout << "MR" << std::endl;

    // CoeffToSlot
    R_Q_square<LOGQ,N> ct_cts[2];
    CoeffToSlot<LOGQ,LOGN,LOGDELTA_cts,G>(ct_modraise,s,ct_cts);
    const int LOGQ_after_cts = LOGQ-(LOGN)/G*LOGDELTA_cts;
    R_Q_square<LOGQ_after_cts,N> ct_ctsrs[2];
    
    #pragma omp parallel for 
    for(int i=0;i<2;i++)
        RS<LOGQ,LOGQ_after_cts,N>(ct_cts[i],ct_ctsrs[i]);

    std::cout << "C2S" << std::endl;

    // EvalMod
    const int LOGQ_after_evalmod = LOGQ_after_cts - 12*LOGDELTA_boot;
    R_Q_square<LOGQ_after_evalmod,N> ct_evalmod[2];
    
    #pragma omp parallel for 
    for(int i=0;i<2;i++)
        EvalMod<LOGQ_after_cts,N,LOGDELTA_boot,K>(ct_ctsrs[i],s,ct_evalmod[i]);

    std::cout << "EvalMod" << std::endl;

    // SlotToCoeff
    const int LOGQ_after_stc = LOGQ_after_evalmod - (LOGN)/G_S*LOGDELTA_stc;
    R_Q_square<LOGQ_after_evalmod,N> ct_stc;
    R_Q_square<LOGQ_after_stc,N> ct_boot;
    SlotToCoeff<LOGQ_after_evalmod,LOGN,LOGDELTA_stc,G_S>(ct_evalmod[0],ct_evalmod[1],s,ct_stc);
    RS<LOGQ_after_evalmod,LOGQ_after_stc,N>(ct_stc,ct_boot);

    std::cout << "S2C" << std::endl;

    R_Q<LOGQ_after_stc,N> pt_out;
    HEAAN<LOGQ_after_stc,N>::dec(ct_boot,s,pt_out);
    decode_log(pt_out,LOGDELTA,z_out);
    print_max_error<LOGN>(z,z_out);

    R_Q<LOGQ_after_stc, N> e;
    resize(pt, e);
    e -= pt_out;

    double e_per_Delta_sup_norm = 0;

    #pragma omp parallel for reduction(max:e_per_Delta_sup_norm) 
    for(int i = 0; i < N; ++i) {
        double val = (double) e[i];
        double val_abs_double = (double) std::abs(val) / Delta;

        e_per_Delta_sup_norm = val_abs_double > e_per_Delta_sup_norm ? val_abs_double : e_per_Delta_sup_norm;
    }

    std::cout << "sup_norm(pt/Delta - pt_tilde/Delta) : " << e_per_Delta_sup_norm << std::endl;
    std::cout << "Modulus consumed : " << (LOGQ - LOGQ_after_stc) << std::endl;
}

template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S>
void EvalRound_bootstrap_test()
{
    std::cout << "Measuring error on proposed bootstrap" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

    const int N = 1 << LOGN;

    int s[N], s_sq[N];
    HEAAN<LOGQ,N>::keygen(H,s);

    Message<LOGN> z, z_out;
    set_test_message(z);

    R_Q<LOGq, N> pt;
    R_Q_square<LOGq,N> ct;
    encode(z,Delta,pt);
    HEAAN<LOGq,N>::enc(pt,s,ct);

	// ModRaise
	R_Q_square<LOGQ,N> ct_modraise;
	mod_raise<LOGq,LOGQ,N>(ct,ct_modraise);

	std::cout << "MR" << std::endl;

	// CoeffToSlot
	R_Q_square<LOGQ,N> ct_cts[2];
	CoeffToSlot<LOGQ,LOGN,LOGDELTA_cts,G>(ct_modraise,s,ct_cts);
	const int LOGQ_after_cts = LOGQ-(LOGN)/G*LOGDELTA_cts;
	R_Q_square<LOGQ_after_cts,N> ct_ctsrs[2];
	for(int i=0;i<2;i++)
	    RS<LOGQ,LOGQ_after_cts,N>(ct_cts[i],ct_ctsrs[i]);

	std::cout << "C2S" << std::endl;

	// EvalMod
	const int LOGQ_after_evalmod = LOGQ_after_cts - 12*LOGDELTA_boot;
	R_Q_square<LOGQ_after_evalmod,N> ct_evalmod[2];
	for(int i=0;i<2;i++)
	    EvalMod<LOGQ_after_cts,N,LOGDELTA_boot,K>(ct_ctsrs[i],s,ct_evalmod[i]);

	std::cout << "EvalMod" << std::endl;
	
	// EvalqI
	R_Q_square<LOGQ_after_evalmod,N> ct_evalqI[2];
	for(int i = 0; i < 2; i++) {
		resize(ct_ctsrs[i], ct_evalqI[i]);
		ct_evalqI[i] -= ct_evalmod[i];
	}

	std::cout << "EvalqI" << std::endl;

	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - (LOGN)/G_S*LOGDELTA_stc;
    R_Q_square<LOGQ_after_evalmod,N> ct_stc;
	R_Q_square<LOGQ_after_stc,N> ct_qI;
	SlotToCoeff<LOGQ_after_evalmod,LOGN,LOGDELTA_stc,G_S>(ct_evalqI[0],ct_evalqI[1],s,ct_stc);
	RS<LOGQ_after_evalmod,LOGQ_after_stc,N>(ct_stc, ct_qI);

	std::cout << "S2C" << std::endl;
	
	// Sub
	R_Q_square<LOGQ_after_stc,N> ct_boot;
	resize(ct_modraise, ct_boot);
	ct_boot -= ct_qI;

	R_Q<LOGQ_after_stc,N> pt_out;
	HEAAN<LOGQ_after_stc,N>::dec(ct_boot,s,pt_out);
	decode_log(pt_out,LOGDELTA,z_out);
	print_max_error<LOGN>(z,z_out);

	R_Q<LOGQ_after_stc, N> e;
	resize(pt, e);
	e -= pt_out;

	double e_per_Delta_sup_norm = 0;
    for(int i = 0; i < N; ++i) {
    	double val = (double) e[i];
		double val_abs_double = (double) std::abs(val) / Delta;

        e_per_Delta_sup_norm = val_abs_double > e_per_Delta_sup_norm ? val_abs_double : e_per_Delta_sup_norm;
    }
	std::cout << "sup_norm(pt/Delta - pt_tilde/Delta) : " << e_per_Delta_sup_norm << std::endl;
	std::cout << "Modulus consumed : " << (LOGQ - LOGQ_after_stc) << std::endl;
}


template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S, int LogDelS>
void EvalRound_modified_bootstrap_test()
{
	std::cout << "Measuring error on EvalRound_modified_bootstrap_test" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

	const int N = 1 << LOGN;
	//const int LogDelS = 30;

	int s[N], s_sq[N];
	HEAAN<LOGQ, N>::keygen(H, s);

	Message<LOGN> z, z_out;
	set_test_message(z);

	R_Q<LOGq, N> pt;
	R_Q_square<LOGq, N> ct;
	encode(z, Delta, pt);
	HEAAN<LOGq, N>::enc(pt, s, ct);

	const int LogQ_new = LOGQ + LogDelS;

	// ModRaise
	R_Q_square<LogQ_new, N> ct_modraise;
	R_Q_square<LOGQ, N> ct_RS;
	mod_raise<LOGq, LogQ_new, N>(ct, ct_modraise);

	RS<LogQ_new, LOGQ, N>(ct_modraise, ct_RS);

	std::cout << "MR" << std::endl;

	// CoeffToSlot
	R_Q_square<LOGQ, N> ct_cts[2];
	CoeffToSlot<LOGQ, LOGN, LOGDELTA_cts, G>(ct_RS, s, ct_cts);
	const int LOGQ_after_cts = LOGQ - (LOGN) / G * LOGDELTA_cts + LogDelS;
	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];
	for (int i = 0; i < 2; i++)
		RS<LOGQ, LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);


	std::cout << "C2S" << std::endl;

	// EvalMod
	const int LOGQ_after_evalmod = LOGQ_after_cts - 12 * LOGDELTA_boot;
	R_Q_square<LOGQ_after_evalmod, N> ct_evalmod[2];
	for (int i = 0; i < 2; i++)
		EvalMod<LOGQ_after_cts, N, LOGDELTA_boot, K>(ct_ctsrs[i], s, ct_evalmod[i]);

	std::cout << "EvalMod" << std::endl;

	// EvalqI
	R_Q_square<LOGQ_after_evalmod, N> ct_evalqI[2];
	for (int i = 0; i < 2; i++) {
		resize(ct_ctsrs[i], ct_evalqI[i]);
		ct_evalqI[i] -= ct_evalmod[i];
	}

	std::cout << "EvalqI" << std::endl;

	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - (LOGN) / G_S * LOGDELTA_stc;
	R_Q_square<LOGQ_after_evalmod, N> ct_stc;
	R_Q_square<LOGQ_after_stc, N> ct_qI;
	SlotToCoeff<LOGQ_after_evalmod, LOGN, LOGDELTA_stc, G_S>(ct_evalqI[0], ct_evalqI[1], s, ct_stc);
	RS<LOGQ_after_evalmod, LOGQ_after_stc, N>(ct_stc, ct_qI);
	std::cout << "S2C" << std::endl;

	// Sub
	R_Q_square<LOGQ_after_stc, N> ct_boot;
	resize(ct_modraise, ct_boot);
	ct_boot -= ct_qI;

	R_Q<LOGQ_after_stc, N> pt_out;
	HEAAN<LOGQ_after_stc, N>::dec(ct_boot, s, pt_out);
	decode_log(pt_out, LOGDELTA, z_out);
	print_max_error<LOGN>(z, z_out);

	R_Q<LOGQ_after_stc, N> e;
	resize(pt, e);
	e -= pt_out;

	double e_per_Delta_sup_norm = 0;
	for (int i = 0; i < N; ++i) {
		double val = (double)e[i];
		double val_abs_double = (double)std::abs(val) / Delta;

		e_per_Delta_sup_norm = val_abs_double > e_per_Delta_sup_norm ? val_abs_double : e_per_Delta_sup_norm;
	}
	std::cout << "sup_norm(pt/Delta - pt_tilde/Delta) : " << e_per_Delta_sup_norm << std::endl;
	std::cout << "Modulus consumed : " << (LOGQ - LOGQ_after_stc) << std::endl;
}



template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S, int LogDelS>
void EvalRound_modified_sw_bootstrap_test()
{
	std::cout << "Measuring error on EvalRound_modified_sw_bootstrap" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

	const int N = 1 << LOGN;
	const int LogQ_new = LOGQ + LogDelS + LOGDELTA_cts;
	const int LogQ_new_rs = LOGQ + LOGDELTA_cts;

	int s[N], s_sq[N];
	HEAAN<LOGQ, N>::keygen(H, s);

	Message<LOGN> z, z_out;
	set_test_message(z);

	R_Q<LOGq, N> pt;
	R_Q_square<LOGq, N> ct;
	encode(z, Delta, pt);
	HEAAN<LOGq, N>::enc(pt, s, ct);

	// ModRaise
	R_Q_square<LogQ_new, N> ct_modraise;
	R_Q_square<LogQ_new_rs, N> ct_RS;
	mod_raise<LOGq, LogQ_new, N>(ct, ct_modraise);

	RS<LogQ_new, LogQ_new_rs, N>(ct_modraise, ct_RS);

	std::cout << "MR" << std::endl;

	// CoeffToSlot
	const int LOGQ_after_cts = LogQ_new_rs - (LOGN) / G * LOGDELTA_cts + LogDelS;
	R_Q_square<LOGQ , N> ct_cts[2];

	CoeffToSlot_sw<LogQ_new_rs, LOGN, LOGDELTA_cts, G>(ct_RS, s, ct_cts);
	
	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];

	#pragma omp parallel for
	for (int i = 0; i < 2; i++)
		RS<LOGQ , LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);

	std::cout << "C2S" << std::endl;

	// EvalMod
	const int LOGQ_after_evalmod = LOGQ_after_cts - 12 * LOGDELTA_boot;
	R_Q_square<LOGQ_after_evalmod, N> ct_evalmod[2];

	#pragma omp parallel for
	for (int i = 0; i < 2; i++)
		EvalMod<LOGQ_after_cts, N, LOGDELTA_boot, K>(ct_ctsrs[i], s, ct_evalmod[i]);

	std::cout << "EvalMod" << std::endl;

	// EvalqI
	R_Q_square<LOGQ_after_evalmod, N> ct_evalqI[2];

	#pragma omp parallel for
	for (int i = 0; i < 2; i++) {
		resize(ct_ctsrs[i], ct_evalqI[i]);
		ct_evalqI[i] -= ct_evalmod[i];
	}

	std::cout << "EvalqI" << std::endl;

	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - (LOGN) / G_S * LOGDELTA_stc;
	R_Q_square<LOGQ_after_evalmod, N> ct_stc;
	R_Q_square<LOGQ_after_stc, N> ct_qI;

	SlotToCoeff<LOGQ_after_evalmod, LOGN, LOGDELTA_stc, G_S>(ct_evalqI[0], ct_evalqI[1], s, ct_stc);
	RS<LOGQ_after_evalmod, LOGQ_after_stc, N>(ct_stc, ct_qI);

	std::cout << "S2C" << std::endl;

	// Sub
	R_Q_square<LOGQ_after_stc, N> ct_boot;
	resize(ct_modraise, ct_boot);
	ct_boot -= ct_qI;

	R_Q<LOGQ_after_stc, N> pt_out;
	HEAAN<LOGQ_after_stc, N>::dec(ct_boot, s, pt_out);
	decode_log(pt_out, LOGDELTA, z_out);
	print_max_error<LOGN>(z, z_out);

	R_Q<LOGQ_after_stc, N> e;
	resize(pt, e);
	e -= pt_out;

	double e_per_Delta_sup_norm = 0;

	#pragma omp parallel for reduction(max:e_per_Delta_sup_norm)
	for (int i = 0; i < N; ++i) {
		double val = (double)e[i];
		double val_abs_double = (double)std::abs(val) / Delta;

		e_per_Delta_sup_norm = val_abs_double > e_per_Delta_sup_norm ? val_abs_double : e_per_Delta_sup_norm;
	}

	std::cout << "sup_norm(pt/Delta - pt_tilde/Delta) : " << e_per_Delta_sup_norm << std::endl;
	std::cout << "Modulus consumed : " << (LOGQ - LOGQ_after_stc) << std::endl;
}


template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S>
void chap3_only_bootstrap_test()
{
	std::cout << "Measuring error on chap3_only_bootstrap" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

	const int N = 1 << LOGN;
	const int LogQ_new = LOGQ + LOGDELTA_cts;

	int s[N], s_sq[N];
	HEAAN<LOGQ, N>::keygen(H, s);

	Message<LOGN> z, z_out;
	set_test_message(z);

	R_Q<LOGq, N> pt;
	R_Q_square<LOGq, N> ct;
	encode(z, Delta, pt);
	HEAAN<LOGq, N>::enc(pt, s, ct);

	// ModRaise
	R_Q_square<LogQ_new, N> ct_modraise;
	mod_raise<LOGq, LogQ_new, N>(ct, ct_modraise);

	std::cout << "MR" << std::endl;

	// CoeffToSlot
	const int LOGQ_after_cts = LogQ_new - (LOGN) / G * LOGDELTA_cts;
	R_Q_square<LOGQ, N> ct_cts[2];

	CoeffToSlot_sw<LogQ_new, LOGN, LOGDELTA_cts, G>(ct_modraise, s, ct_cts);

	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];
	for (int i = 0; i < 2; i++)
		RS<LOGQ, LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);


	std::cout << "C2S" << std::endl;

	// EvalMod
	const int LOGQ_after_evalmod = LOGQ_after_cts - 12 * LOGDELTA_boot;
	R_Q_square<LOGQ_after_evalmod, N> ct_evalmod[2];
	for (int i = 0; i < 2; i++)
		EvalMod<LOGQ_after_cts, N, LOGDELTA_boot, K>(ct_ctsrs[i], s, ct_evalmod[i]);

	std::cout << "EvalMod" << std::endl;

	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - (LOGN) / G_S * LOGDELTA_stc;
	R_Q_square<LOGQ_after_evalmod, N> ct_stc;
	R_Q_square<LOGQ_after_stc, N> ct_boot;
	SlotToCoeff<LOGQ_after_evalmod, LOGN, LOGDELTA_stc, G_S>(ct_evalmod[0], ct_evalmod[1], s, ct_stc);
	RS<LOGQ_after_evalmod, LOGQ_after_stc, N>(ct_stc, ct_boot);


	std::cout << "S2C" << std::endl;


	R_Q<LOGQ_after_stc, N> pt_out;
	HEAAN<LOGQ_after_stc, N>::dec(ct_boot, s, pt_out);
	decode_log(pt_out, LOGDELTA, z_out);
	print_max_error<LOGN>(z, z_out);

	R_Q<LOGQ_after_stc, N> e;
	resize(pt, e);
	e -= pt_out;

	double e_per_Delta_sup_norm = 0;
	for (int i = 0; i < N; ++i) {
		double val = (double)e[i];
		double val_abs_double = (double)std::abs(val) / Delta;

		e_per_Delta_sup_norm = val_abs_double > e_per_Delta_sup_norm ? val_abs_double : e_per_Delta_sup_norm;
	}
	std::cout << "sup_norm(pt/Delta - pt_tilde/Delta) : " << e_per_Delta_sup_norm << std::endl;
	std::cout << "Modulus consumed : " << (LOGQ - LOGQ_after_stc) << std::endl;
}

template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S2C, int LOG_qdiv, int LOG_ql, int LogDelS>
void all_together_bootstrap_test()
{
	std::cout << "Measuring error on all together bootstrap" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

	const int N = 1 << LOGN;
	const int LogQ_new = LOGQ + LogDelS + LOGDELTA_cts;
	const int LogQ_new_rs = LOGQ + LOGDELTA_cts;

	int s[N], s_sq[N];
	HEAAN<LOGQ, N>::keygen(H, s);

	Message<LOGN> z, z_out;
	set_test_message(z);

	R_Q<LOGq, N> pt;
	R_Q_square<LOGq, N> ct;
	encode(z, Delta, pt);
	HEAAN<LOGq, N>::enc(pt, s, ct);

	// ModRaise
	R_Q_square<LogQ_new, N> ct_modraise;
	R_Q_square<LogQ_new_rs, N> ct_RS;
	mod_raise<LOGq, LogQ_new, N>(ct, ct_modraise);

	RS<LogQ_new, LogQ_new_rs, N>(ct_modraise, ct_RS);

	std::cout << "MR" << std::endl;

	// CoeffToSlot
	const int LOGQ_after_cts = LogQ_new_rs - (LOGN) / G * LOGDELTA_cts + LogDelS;
	R_Q_square<LOGQ, N> ct_cts[2];

	CoeffToSlot_sw<LogQ_new_rs, LOGN, LOGDELTA_cts, G>(ct_RS, s, ct_cts);

	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];

	#pragma omp parallel for
	for (int i = 0; i < 2; i++)
		RS<LOGQ, LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);

	std::cout << "C2S" << std::endl;

	// EvalMod
	const int LOGQ_after_evalmod = LOGQ_after_cts - 12 * LOGDELTA_boot;
	R_Q_square<LOGQ_after_evalmod, N> ct_evalmod[2];

	#pragma omp parallel for
	for (int i = 0; i < 2; i++)
		EvalMod<LOGQ_after_cts, N, LOGDELTA_boot, K>(ct_ctsrs[i], s, ct_evalmod[i]);

	std::cout << "EvalMod" << std::endl;

	// EvalqI
	R_Q_square<LOGQ_after_evalmod, N> ct_evalqI[2];

	#pragma omp parallel for
	for (int i = 0; i < 2; i++) {
		resize(ct_ctsrs[i], ct_evalqI[i]);
		ct_evalqI[i] -= ct_evalmod[i];
	}

	std::cout << "EvalqI" << std::endl;

	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - LOG_qdiv - (LOGN) / G_S2C * LOG_ql;
	R_Q_square<LOGQ_after_evalmod - LOG_qdiv, N> ct_stc, ct_stc_up, ct_stc_down;
	R_Q_square<LOGQ_after_stc, N> ct_qI;
	R_Q_square<LOGQ_after_evalmod - LOG_qdiv, N> ct_evalqI0_up, ct_evalqI0_down, ct_evalqI1_up, ct_evalqI1_down;

	#pragma omp parallel sections
	{
		#pragma omp section
		decomp< LOGQ_after_evalmod , N, LOG_qdiv>(ct_evalqI[0], ct_evalqI0_up, ct_evalqI0_down);
		
		#pragma omp section
		decomp< LOGQ_after_evalmod , N, LOG_qdiv>(ct_evalqI[1], ct_evalqI1_up, ct_evalqI1_down);
	}

	tuple_SlotToCoeff<LOGQ_after_evalmod - LOG_qdiv, LOGN, LOGDELTA_stc, G_S2C, LOG_qdiv>(ct_evalqI0_up, ct_evalqI0_down, ct_evalqI1_up, ct_evalqI1_down, s, ct_stc_up, ct_stc_down);
	recomb< LOGQ_after_evalmod - LOG_qdiv, N, LOG_qdiv>(ct_stc_up, ct_stc_down, ct_stc);

	RS<LOGQ_after_evalmod - LOG_qdiv, LOGQ_after_stc, N>(ct_stc, ct_qI);

	std::cout << "S2C" << std::endl;

	// Sub
	R_Q_square<LOGQ_after_stc, N> ct_boot;
	resize(ct_modraise, ct_boot);
	ct_boot -= ct_qI;

	R_Q<LOGQ_after_stc, N> pt_out;
	HEAAN<LOGQ_after_stc, N>::dec(ct_boot, s, pt_out);
	decode_log(pt_out, LOGDELTA, z_out);
	print_max_error<LOGN>(z, z_out);

	R_Q<LOGQ_after_stc, N> e;
	resize(pt, e);
	e -= pt_out;

	double e_per_Delta_sup_norm = 0;

	#pragma omp parallel for reduction(max:e_per_Delta_sup_norm)
	for (int i = 0; i < N; ++i) {
		double val = (double)e[i];
		double val_abs_double = (double)std::abs(val) / Delta;

		e_per_Delta_sup_norm = val_abs_double > e_per_Delta_sup_norm ? val_abs_double : e_per_Delta_sup_norm;
	}

	std::cout << "sup_norm(pt/Delta - pt_tilde/Delta) : " << e_per_Delta_sup_norm << std::endl;
	std::cout << "Modulus consumed : " << (LOGQ - LOGQ_after_stc) << std::endl;
}


template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S, int LOG_qdiv_C, int LOG_ql_C, int LOG_qdiv, int LOG_ql>
void tuple_bootstrap_test()
{
	std::cout << "Measuring error on tuple bootstrap" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

	const int N = 1 << LOGN;

	int s[N], s_sq[N];
	HEAAN<LOGQ, N>::keygen(H, s);

	Message<LOGN> z, z_out;
	set_test_message(z);

	R_Q<LOGq, N> pt;
	R_Q_square<LOGq, N> ct;
	encode(z, Delta, pt);
	HEAAN<LOGq, N>::enc(pt, s, ct);

	// ModRaise
	R_Q_square<LOGQ, N> ct_modraise;
	mod_raise<LOGq, LOGQ, N>(ct, ct_modraise);

	std::cout << "MR" << std::endl;

	// CoeffToSlot
	const int LOGQ_after_cts = LOGQ - LOG_qdiv_C - (LOGN) / G * LOG_ql_C;

	R_Q_square<LOGQ - LOG_qdiv_C, N> ct_cts[2], ctstemp_up, ctstemp_down;

	decomp< LOGQ, N, LOG_qdiv_C>(ct_modraise, ctstemp_up, ctstemp_down);

	
	tuple_CoeffToSlot<LOGQ - LOG_qdiv_C, LOGN, LOGDELTA_cts, G, LOG_qdiv_C>(ctstemp_up, ctstemp_down, s, ct_cts);

	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];
	for (int i = 0; i < 2; i++)
		RS<LOGQ - LOG_qdiv_C, LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);


	std::cout << "C2S" << std::endl;

	// EvalMod
	const int LOGQ_after_evalmod = LOGQ_after_cts - 12 * LOGDELTA_boot;
	R_Q_square<LOGQ_after_evalmod, N> ct_evalmod[2];
	for (int i = 0; i < 2; i++)
		EvalMod<LOGQ_after_cts, N, LOGDELTA_boot, K>(ct_ctsrs[i], s, ct_evalmod[i]);

	std::cout << "EvalMod" << std::endl;

	// EvalqI
	R_Q_square<LOGQ_after_evalmod, N> ct_evalqI[2];
	for (int i = 0; i < 2; i++) {
		resize(ct_ctsrs[i], ct_evalqI[i]);
		ct_evalqI[i] -= ct_evalmod[i];
	}

	std::cout << "EvalqI" << std::endl;

	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - LOG_qdiv - (LOGN) / G_S * LOG_ql;
	R_Q_square<LOGQ_after_evalmod - LOG_qdiv, N> ct_stc, ct_stc_up, ct_stc_down;
	R_Q_square<LOGQ_after_stc, N> ct_qI;
	R_Q_square<LOGQ_after_evalmod - LOG_qdiv, N> ct_evalqI0_up, ct_evalqI0_down, ct_evalqI1_up, ct_evalqI1_down;
	decomp< LOGQ_after_evalmod, N, LOG_qdiv>(ct_evalqI[0], ct_evalqI0_up, ct_evalqI0_down);
	decomp< LOGQ_after_evalmod, N, LOG_qdiv>(ct_evalqI[1], ct_evalqI1_up, ct_evalqI1_down);

	tuple_SlotToCoeff<LOGQ_after_evalmod - LOG_qdiv, LOGN, LOGDELTA_stc, G_S, LOG_qdiv>(ct_evalqI0_up, ct_evalqI0_down, ct_evalqI1_up, ct_evalqI1_down, s, ct_stc_up, ct_stc_down);
	recomb< LOGQ_after_evalmod - LOG_qdiv, N, LOG_qdiv>(ct_stc_up, ct_stc_down, ct_stc);

	RS<LOGQ_after_evalmod - LOG_qdiv, LOGQ_after_stc, N>(ct_stc, ct_qI);


	std::cout << "S2C" << std::endl;

	// Sub
	R_Q_square<LOGQ_after_stc, N> ct_boot;
	resize(ct_modraise, ct_boot);
	ct_boot -= ct_qI;

	R_Q<LOGQ_after_stc, N> pt_out;
	HEAAN<LOGQ_after_stc, N>::dec(ct_boot, s, pt_out);
	decode_log(pt_out, LOGDELTA, z_out);
	print_max_error<LOGN>(z, z_out);

	R_Q<LOGQ_after_stc, N> e;
	resize(pt, e);
	e -= pt_out;

	double e_per_Delta_sup_norm = 0;
	for (int i = 0; i < N; ++i) {
		double val = (double)e[i];
		double val_abs_double = (double)std::abs(val) / Delta;

		e_per_Delta_sup_norm = val_abs_double > e_per_Delta_sup_norm ? val_abs_double : e_per_Delta_sup_norm;
	}
	std::cout << "sup_norm(pt/Delta - pt_tilde/Delta) : " << e_per_Delta_sup_norm << std::endl;
	std::cout << "Modulus consumed : " << (LOGQ - LOGQ_after_stc) << std::endl;
}


template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S, int LOG_qdiv, int LOG_ql, int LogDelS, int LOGDELTA_ori>
void erpluspar_all_together_bootstrap_test()
{
	std::cout << "Measuring error on all together bootstrap w erpluspar" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

	const int N = 1 << LOGN;
	const int LogQ_new = LOGQ + LogDelS + LOGDELTA_cts;
	const int LogQ_new_rs = LOGQ + LOGDELTA_cts;

	int s[N], s_sq[N];
	HEAAN<LOGQ, N>::keygen(H, s);

	Message<LOGN> z, z_out;
	set_test_message(z);

	R_Q<LOGq, N> pt;
	R_Q_square<LOGq, N> ct;
	encode(z, Delta, pt);
	HEAAN<LOGq, N>::enc(pt, s, ct);

	// ModRaise
	R_Q_square<LogQ_new, N> ct_modraise;
	R_Q_square<LogQ_new_rs, N> ct_RS;
	mod_raise<LOGq, LogQ_new, N>(ct, ct_modraise);

	RS<LogQ_new, LogQ_new_rs, N>(ct_modraise, ct_RS);

	std::cout << "MR" << std::endl;

	auto start_1 = std::chrono::high_resolution_clock::now();


	const int LOGQ_after_cts = LogQ_new_rs - (LOGN) / G * LOGDELTA_cts + LogDelS;
	R_Q_square<LOGQ, N> ct_cts[2];
	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];
	const int LOGQ_after_evalmod = LOGQ_after_cts - 12 * LOGDELTA_boot;
	R_Q_square<LOGQ_after_evalmod, N> ct_evalmod[2];
	R_Q_square<LOGQ_after_evalmod, N> ct_evalqI[2];
	const int LOG_par = LOGQ_after_evalmod + (LOGN) / G * LOGDELTA_ori;
	R_Q_square<LOG_par, N> ct_cts_par_before;
	R_Q_square<LOG_par, N> ct_cts_par[2];
	R_Q_square<LOGQ_after_evalmod, N> ct_par_res[2];


#pragma omp parallel sections
	{
#pragma omp section
		{

			CoeffToSlot_sw<LogQ_new_rs, LOGN, LOGDELTA_cts, G>(ct_RS, s, ct_cts);


#pragma omp parallel for
			for (int i = 0; i < 2; i++)
				RS<LOGQ, LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);

			std::cout << "C2S" << std::endl;

			

#pragma omp parallel for
			for (int i = 0; i < 2; i++)
				EvalMod<LOGQ_after_cts, N, LOGDELTA_boot, K>(ct_ctsrs[i], s, ct_evalmod[i]);

			std::cout << "EvalMod" << std::endl;

			

#pragma omp parallel for
			for (int i = 0; i < 2; i++) {
				resize(ct_ctsrs[i], ct_evalqI[i]);
				ct_evalqI[i] -= ct_evalmod[i];
			}

			std::cout << "EvalqI" << std::endl;
		}

#pragma omp section
		{
			
			resize(ct_modraise, ct_cts_par_before);

			
			CoeffToSlot<LOG_par, LOGN, LOGDELTA_ori, G>(ct_cts_par_before, s, ct_cts_par);

			
			for (int i = 0; i < 2; i++)
				RS<LOG_par, LOGQ_after_evalmod, N>(ct_cts_par[i], ct_par_res[i]);

			std::cout << "parC2S" << std::endl;
		}
	}


	for (int i = 0; i < 2; i++) {
		ct_par_res[i] -= ct_evalqI[i];
	}

	std::cout << "comb" << std::endl;

	auto end_1 = std::chrono::high_resolution_clock::now();
    	auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1);
    	std::cout << "Time taken for C2S Evalmod section: " << duration_1.count()/1000 << " s" << std::endl;


	auto start_2 = std::chrono::high_resolution_clock::now();



	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - LOG_qdiv - (LOGN) / G_S * LOG_ql;
	R_Q_square<LOGQ_after_evalmod - LOG_qdiv, N> ct_stc, ct_stc_up, ct_stc_down;
	R_Q_square<LOGQ_after_stc, N> ct_boot;
	R_Q_square<LOGQ_after_evalmod - LOG_qdiv, N> ct_par_res0_up, ct_par_res0_down, ct_par_res1_up, ct_par_res1_down;

#pragma omp parallel sections
	{
#pragma omp section
		decomp< LOGQ_after_evalmod, N, LOG_qdiv>(ct_par_res[0], ct_par_res0_up, ct_par_res0_down);

#pragma omp section
		decomp< LOGQ_after_evalmod, N, LOG_qdiv>(ct_par_res[1], ct_par_res1_up, ct_par_res1_down);
	}

	tuple_SlotToCoeff<LOGQ_after_evalmod - LOG_qdiv, LOGN, LOGDELTA_stc, G_S, LOG_qdiv>(ct_par_res0_up, ct_par_res0_down, ct_par_res1_up, ct_par_res1_down, s, ct_stc_up, ct_stc_down);
	recomb< LOGQ_after_evalmod - LOG_qdiv, N, LOG_qdiv>(ct_stc_up, ct_stc_down, ct_stc);

	RS<LOGQ_after_evalmod - LOG_qdiv, LOGQ_after_stc, N>(ct_stc, ct_boot);

	std::cout << "S2C" << std::endl;

	auto end_2 = std::chrono::high_resolution_clock::now();
    	auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - start_2);
    	std::cout << "Time taken for S2C section: " << duration_2.count() / 1000 << " s" << std::endl;



	R_Q<LOGQ_after_stc, N> pt_out;
	HEAAN<LOGQ_after_stc, N>::dec(ct_boot, s, pt_out);
	decode_log(pt_out, LOGDELTA, z_out);
	print_max_error<LOGN>(z, z_out);

	R_Q<LOGQ_after_stc, N> e;
	resize(pt, e);
	e -= pt_out;

	double e_per_Delta_sup_norm = 0;

#pragma omp parallel for reduction(max:e_per_Delta_sup_norm)
	for (int i = 0; i < N; ++i) {
		double val = (double)e[i];
		double val_abs_double = (double)std::abs(val) / Delta;

		e_per_Delta_sup_norm = val_abs_double > e_per_Delta_sup_norm ? val_abs_double : e_per_Delta_sup_norm;
	}

	std::cout << "sup_norm(pt/Delta - pt_tilde/Delta) : " << e_per_Delta_sup_norm << std::endl;
	std::cout << "Modulus consumed : " << (LOGQ - LOGQ_after_stc) << std::endl;
}





template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S, int LOGDELTA_ori>
void evalroundplus_test()
{
	std::cout << "Measuring error on evalroundplus" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;


	const int N = 1 << LOGN;

	int s[N], s_sq[N];
	HEAAN<LOGQ, N>::keygen(H, s);

	Message<LOGN> z, z_out;
	set_test_message(z);

	R_Q<LOGq, N> pt;
	R_Q_square<LOGq, N> ct;
	encode(z, Delta, pt);
	HEAAN<LOGq, N>::enc(pt, s, ct);

	// ModRaise
	R_Q_square<LOGQ, N> ct_modraise;
	mod_raise<LOGq, LOGQ, N>(ct, ct_modraise);

	std::cout << "MR" << std::endl;


	const int LOGQ_after_cts = LOGQ - (LOGN) / G * LOGDELTA_cts;
	R_Q_square<LOGQ, N> ct_cts[2];
	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];
	const int LOGQ_after_evalmod = LOGQ_after_cts - 12 * LOGDELTA_boot;
	R_Q_square<LOGQ_after_evalmod, N> ct_evalmod[2];
	R_Q_square<LOGQ_after_evalmod, N> ct_evalqI[2];
	const int LOG_par = LOGQ_after_evalmod + (LOGN) / G * LOGDELTA_ori;
	R_Q_square<LOG_par, N> ct_cts_par_before;
	R_Q_square<LOG_par, N> ct_cts_par[2];
	R_Q_square<LOGQ_after_evalmod, N> ct_par_res[2];

	//check

	auto start_1 = std::chrono::high_resolution_clock::now();



#pragma omp parallel sections
	{
#pragma omp section
		{

			CoeffToSlot<LOGQ, LOGN, LOGDELTA_cts, G>(ct_modraise, s, ct_cts);


#pragma omp parallel for
			for (int i = 0; i < 2; i++)
				RS<LOGQ, LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);

			std::cout << "C2S" << std::endl;



#pragma omp parallel for
			for (int i = 0; i < 2; i++)
				EvalMod<LOGQ_after_cts, N, LOGDELTA_boot, K>(ct_ctsrs[i], s, ct_evalmod[i]);

			std::cout << "EvalMod" << std::endl;



#pragma omp parallel for
			for (int i = 0; i < 2; i++) {
				resize(ct_ctsrs[i], ct_evalqI[i]);
				ct_evalqI[i] -= ct_evalmod[i];
			}

			std::cout << "EvalqI" << std::endl;
		}

#pragma omp section
		{

			resize(ct_modraise, ct_cts_par_before);


			CoeffToSlot<LOG_par, LOGN, LOGDELTA_ori, G>(ct_cts_par_before, s, ct_cts_par);


			for (int i = 0; i < 2; i++)
				RS<LOG_par, LOGQ_after_evalmod, N>(ct_cts_par[i], ct_par_res[i]);

			std::cout << "par_C2S" << std::endl;
		}
	}


	for (int i = 0; i < 2; i++) {
		ct_par_res[i] -= ct_evalqI[i];
	}

	std::cout << "comb" << std::endl;

	auto end_1 = std::chrono::high_resolution_clock::now();
    	auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1);
    	std::cout << "Time taken for C2S Evalmod section: " << duration_1.count()/1000 << " s" << std::endl;


	auto start_2 = std::chrono::high_resolution_clock::now();



	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - (LOGN) / G_S * LOGDELTA_stc;
	R_Q_square<LOGQ_after_evalmod, N> ct_stc;
	R_Q_square<LOGQ_after_stc, N> ct_boot;
	SlotToCoeff<LOGQ_after_evalmod, LOGN, LOGDELTA_stc, G_S>(ct_par_res[0], ct_par_res[1], s, ct_stc);
	RS<LOGQ_after_evalmod, LOGQ_after_stc, N>(ct_stc, ct_boot);

	std::cout << "S2C" << std::endl;

	auto end_2 = std::chrono::high_resolution_clock::now();
    	auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - start_2);
    	std::cout << "Time taken for S2C section: " << duration_2.count() / 1000 << " s" << std::endl;



	R_Q<LOGQ_after_stc, N> pt_out;
	HEAAN<LOGQ_after_stc, N>::dec(ct_boot, s, pt_out);
	decode_log(pt_out, LOGDELTA, z_out);
	print_max_error<LOGN>(z, z_out);

	R_Q<LOGQ_after_stc, N> e;
	resize(pt, e);
	e -= pt_out;

	double e_per_Delta_sup_norm = 0;
	for (int i = 0; i < N; ++i) {
		double val = (double)e[i];
		double val_abs_double = (double)std::abs(val) / Delta;

		e_per_Delta_sup_norm = val_abs_double > e_per_Delta_sup_norm ? val_abs_double : e_per_Delta_sup_norm;
	}
	std::cout << "sup_norm(pt/Delta - pt_tilde/Delta) : " << e_per_Delta_sup_norm << std::endl;
	std::cout << "Modulus consumed : " << (LOGQ - LOGQ_after_stc) << std::endl;

}

template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S, int LOGDELTA_ori>
void evalroundplus_test_S2C_first()
{
	std::cout << "Measuring error on evalroundplus_S2C_first" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;


	const int N = 1 << LOGN;

	int s[N], s_sq[N];
	HEAAN<LOGQ, N>::keygen(H, s);

	const int LOGq_up = LOGq + (LOGN) / G_S * LOGDELTA_stc;

	int s_conj[N];
	R_Q_square<2 * LOGq_up, 1 << LOGN> ckey;
	conj<N>(s, s_conj);
	HEAAN<LOGq_up, N>::swkgen(s_conj, s, ckey);

	Message<LOGN> z, z_out;
	set_test_message(z);


	R_Q<LOGq_up, N> pt;
	R_Q_square<LOGq_up, N> ct;
	R_Q_square<LOGq_up, N> ct_conj;
	encode(z, Delta, pt);
	HEAAN<LOGq_up, N>::enc(pt, s, ct);

	conj(ct, ckey, ct_conj);

	R_Q_square<LOGq_up, N> ct_1;
	R_Q_square<LOGq_up, N> ct_2;
	R_Q_square<LOGq_up - 1, N> ct_1_d;
	R_Q_square<LOGq_up - 1, N> ct_2_d;

	ct_1 = ct;
	ct_2 = ct;
		
	ct_2 -= ct;

	R_Q_square<LOGq_up, N> ct_stc;

	SlotToCoeff<LOGq_up, LOGN, LOGDELTA_stc, G_S>(ct_1, ct_2, s, ct_stc);

	R_Q_square<LOGq, N> ct_base;

	RS<LOGq_up, LOGq, N>(ct_stc, ct_base);

	
	std::cout << "S2C" << std::endl;
	

	// ModRaise
	R_Q_square<LOGQ, N> ct_modraise;
	mod_raise<LOGq, LOGQ, N>(ct_base, ct_modraise);

	std::cout << "MR" << std::endl;


	const int LOGQ_after_cts = LOGQ - (LOGN) / G * LOGDELTA_cts;
	R_Q_square<LOGQ, N> ct_cts[2];
	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];
	const int LOGQ_after_evalmod = LOGQ_after_cts - 12 * LOGDELTA_boot;
	R_Q_square<LOGQ_after_evalmod, N> ct_evalmod[2];
	R_Q_square<LOGQ_after_evalmod, N> ct_evalqI[2];
	const int LOG_par = LOGQ_after_evalmod + (LOGN) / G * LOGDELTA_ori;
	R_Q_square<LOG_par, N> ct_cts_par_before;
	R_Q_square<LOG_par, N> ct_cts_par[2];
	R_Q_square<LOGQ_after_evalmod, N> ct_par_res[2];


	auto start_1 = std::chrono::high_resolution_clock::now();



#pragma omp parallel sections
	{
#pragma omp section
		{

			CoeffToSlot<LOGQ, LOGN, LOGDELTA_cts, G>(ct_modraise, s, ct_cts);


#pragma omp parallel for
			for (int i = 0; i < 2; i++)
				RS<LOGQ, LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);

			std::cout << "C2S" << std::endl;



#pragma omp parallel for
			for (int i = 0; i < 2; i++)
				EvalMod<LOGQ_after_cts, N, LOGDELTA_boot, K>(ct_ctsrs[i], s, ct_evalmod[i]);

			std::cout << "EvalMod" << std::endl;



#pragma omp parallel for
			for (int i = 0; i < 2; i++) {
				resize(ct_ctsrs[i], ct_evalqI[i]);
				ct_evalqI[i] -= ct_evalmod[i];
			}

			std::cout << "EvalqI" << std::endl;
		}

#pragma omp section
		{

			resize(ct_modraise, ct_cts_par_before);


			CoeffToSlot<LOG_par, LOGN, LOGDELTA_ori, G>(ct_cts_par_before, s, ct_cts_par);


			for (int i = 0; i < 2; i++)
				RS<LOG_par, LOGQ_after_evalmod, N>(ct_cts_par[i], ct_par_res[i]);

			std::cout << "par_C2S" << std::endl;
		}
	}


	for (int i = 0; i < 2; i++) {
		ct_par_res[i] -= ct_evalqI[i];
	}

	std::cout << "comb" << std::endl;

	auto end_1 = std::chrono::high_resolution_clock::now();
	auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1);
	std::cout << "Time taken for C2S Evalmod section: " << duration_1.count() / 1000 << " s" << std::endl;


	//check

	auto start_2 = std::chrono::high_resolution_clock::now();



	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - (LOGN) / G_S * LOGDELTA_stc;
	R_Q_square<LOGQ_after_evalmod, N> ct_boot;

	R_Q_square<LOGQ_after_evalmod, N> ct_conj_2;

	R_Q_square<2 * LOGQ_after_evalmod, 1 << LOGN> ckey2;
	HEAAN<LOGQ_after_evalmod, N>::swkgen(s_conj, s, ckey2);

	conj(ct_par_res[1], ckey2, ct_conj_2);

	ct_par_res[0] += ct_conj_2;


	auto end_2 = std::chrono::high_resolution_clock::now();
	auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - start_2);
	std::cout << "Time taken for S2C section: " << duration_2.count() / 1000 << " s" << std::endl;



	R_Q<LOGQ_after_evalmod, N> pt_out;
	HEAAN<LOGQ_after_evalmod, N>::dec(ct_par_res[0], s, pt_out);
	decode_log(pt_out, LOGDELTA, z_out);
	print_max_error<LOGN>(z, z_out);

	R_Q<LOGQ_after_evalmod, N> e;
	resize(pt, e);
	e -= pt_out;

	double e_per_Delta_sup_norm = 0;
	for (int i = 0; i < N; ++i) {
		double val = (double)e[i];
		double val_abs_double = (double)std::abs(val) / Delta;

		e_per_Delta_sup_norm = val_abs_double > e_per_Delta_sup_norm ? val_abs_double : e_per_Delta_sup_norm;
	}
	std::cout << "sup_norm(pt/Delta - pt_tilde/Delta) : " << e_per_Delta_sup_norm << std::endl;
	std::cout << "Modulus consumed : " << (LOGQ - LOGQ_after_stc) << std::endl;

}



template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S, int LOG_qdiv, int LOG_ql, int LogDelS, int LOGDELTA_ori>
void erpluspar_12_S2C_first()
{
	std::cout << "Measuring error on erpluspar_S2C_first_12" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

	const int N = 1 << LOGN;
	const int LogQ_new = LOGQ + LogDelS + LOGDELTA_cts;
	const int LogQ_new_rs = LOGQ + LOGDELTA_cts;

	int s[N], s_sq[N];
	HEAAN<LOGQ, N>::keygen(H, s);



	const int LOGq_up = LOGq + (LOGN) / G_S * LOGDELTA_stc;

	int s_conj[N];
	R_Q_square<2 * LOGq_up, 1 << LOGN> ckey;
	conj<N>(s, s_conj);
	HEAAN<LOGq_up, N>::swkgen(s_conj, s, ckey);

	Message<LOGN> z, z_out;
	set_test_message(z);


	R_Q<LOGq_up, N> pt;
	R_Q_square<LOGq_up, N> ct;
	R_Q_square<LOGq_up, N> ct_conj;
	encode(z, Delta, pt);
	HEAAN<LOGq_up, N>::enc(pt, s, ct);

	conj(ct, ckey, ct_conj);

	R_Q_square<LOGq_up, N> ct_1;
	R_Q_square<LOGq_up, N> ct_2;
	R_Q_square<LOGq_up - 1, N> ct_1_d;
	R_Q_square<LOGq_up - 1, N> ct_2_d;

	ct_1 = ct;
	ct_2 = ct;

	ct_2 -= ct;

	R_Q_square<LOGq_up, N> ct_stc;

	SlotToCoeff<LOGq_up, LOGN, LOGDELTA_stc, G_S>(ct_1, ct_2, s, ct_stc);

	R_Q_square<LOGq, N> ct_base;

	RS<LOGq_up, LOGq, N>(ct_stc, ct_base);


	std::cout << "S2C" << std::endl;


	// ModRaise
	R_Q_square<LogQ_new, N> ct_modraise;
	R_Q_square<LogQ_new_rs, N> ct_RS;
	mod_raise<LOGq, LogQ_new, N>(ct_base, ct_modraise);

	RS<LogQ_new, LogQ_new_rs, N>(ct_modraise, ct_RS);

	std::cout << "MR" << std::endl;

	auto start_1 = std::chrono::high_resolution_clock::now();


	const int LOGQ_after_cts = LogQ_new_rs - (LOGN) / G * LOGDELTA_cts + LogDelS;
	R_Q_square<LOGQ, N> ct_cts[2];
	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];
	const int LOGQ_after_evalmod = LOGQ_after_cts - 12 * LOGDELTA_boot;
	R_Q_square<LOGQ_after_evalmod, N> ct_evalmod[2];
	R_Q_square<LOGQ_after_evalmod, N> ct_evalqI[2];
	const int LOG_par = LOGQ_after_evalmod + (LOGN) / G * LOGDELTA_ori;
	R_Q_square<LOG_par, N> ct_cts_par_before;
	R_Q_square<LOG_par, N> ct_cts_par[2];
	R_Q_square<LOGQ_after_evalmod, N> ct_par_res[2];


#pragma omp parallel sections
	{
#pragma omp section
		{

			CoeffToSlot_sw<LogQ_new_rs, LOGN, LOGDELTA_cts, G>(ct_RS, s, ct_cts);


#pragma omp parallel for
			for (int i = 0; i < 2; i++)
				RS<LOGQ, LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);

			std::cout << "C2S" << std::endl;



#pragma omp parallel for
			for (int i = 0; i < 2; i++)
				EvalMod<LOGQ_after_cts, N, LOGDELTA_boot, K>(ct_ctsrs[i], s, ct_evalmod[i]);

			std::cout << "EvalMod" << std::endl;



#pragma omp parallel for
			for (int i = 0; i < 2; i++) {
				resize(ct_ctsrs[i], ct_evalqI[i]);
				ct_evalqI[i] -= ct_evalmod[i];
			}

			std::cout << "EvalqI" << std::endl;
		}

#pragma omp section
		{

			resize(ct_modraise, ct_cts_par_before);


			CoeffToSlot<LOG_par, LOGN, LOGDELTA_ori, G>(ct_cts_par_before, s, ct_cts_par);


			for (int i = 0; i < 2; i++)
				RS<LOG_par, LOGQ_after_evalmod, N>(ct_cts_par[i], ct_par_res[i]);

			std::cout << "parC2S" << std::endl;
		}
	}


	for (int i = 0; i < 2; i++) {
		ct_par_res[i] -= ct_evalqI[i];
	}

	std::cout << "comb" << std::endl;

	auto end_1 = std::chrono::high_resolution_clock::now();
	auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1);
	std::cout << "Time taken for C2S Evalmod section: " << duration_1.count() / 1000 << " s" << std::endl;


	auto start_2 = std::chrono::high_resolution_clock::now();


	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - (LOGN) / G_S * LOGDELTA_stc;

	R_Q_square<LOGQ_after_evalmod, N> ct_boot;

	R_Q_square<LOGQ_after_evalmod, N> ct_conj_2;

	R_Q_square<2 * LOGQ_after_evalmod, 1 << LOGN> ckey2;
	HEAAN<LOGQ_after_evalmod, N>::swkgen(s_conj, s, ckey2);

	conj(ct_par_res[1], ckey2, ct_conj_2);

	ct_par_res[0] += ct_conj_2;


	auto end_2 = std::chrono::high_resolution_clock::now();
	auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - start_2);
	std::cout << "Time taken for S2C section: " << duration_2.count() / 1000 << " s" << std::endl;



	R_Q<LOGQ_after_evalmod, N> pt_out;
	HEAAN<LOGQ_after_evalmod, N>::dec(ct_par_res[0], s, pt_out);
	decode_log(pt_out, LOGDELTA, z_out);
	print_max_error<LOGN>(z, z_out);

	R_Q<LOGQ_after_evalmod, N> e;
	resize(pt, e);
	e -= pt_out;

	double e_per_Delta_sup_norm = 0;

#pragma omp parallel for reduction(max:e_per_Delta_sup_norm)
	for (int i = 0; i < N; ++i) {
		double val = (double)e[i];
		double val_abs_double = (double)std::abs(val) / Delta;

		e_per_Delta_sup_norm = val_abs_double > e_per_Delta_sup_norm ? val_abs_double : e_per_Delta_sup_norm;
	}

	std::cout << "sup_norm(pt/Delta - pt_tilde/Delta) : " << e_per_Delta_sup_norm << std::endl;
	std::cout << "Modulus consumed : " << (LOGQ - LOGQ_after_stc) << std::endl;
}



template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S, int LOG_qdiv, int LOG_ql, int LogDelS, int LOGDELTA_ori>
void erpluspar_all_together_bootstrap_test_S2C_first()
{
	std::cout << "Measuring error on all together bootstrap w erpluspar_S2C_first" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

	const int N = 1 << LOGN;
	const int LogQ_new = LOGQ + LogDelS + LOGDELTA_cts;
	const int LogQ_new_rs = LOGQ + LOGDELTA_cts;

	int s[N], s_sq[N];
	HEAAN<LOGQ, N>::keygen(H, s);



	const int LOGq_up = LOGq + LOG_qdiv + (LOGN) / G_S * LOG_ql;

	int s_conj[N];
	R_Q_square<2 * LOGq_up, 1 << LOGN> ckey;
	conj<N>(s, s_conj);
	HEAAN<LOGq_up, N>::swkgen(s_conj, s, ckey);

	Message<LOGN> z, z_out;
	set_test_message(z);


	R_Q<LOGq_up, N> pt;
	R_Q_square<LOGq_up, N> ct;
	R_Q_square<LOGq_up, N> ct_conj;
	encode(z, Delta, pt);
	HEAAN<LOGq_up, N>::enc(pt, s, ct);

	conj(ct, ckey, ct_conj);

	R_Q_square<LOGq_up, N> ct_1;
	R_Q_square<LOGq_up, N> ct_2;
	R_Q_square<LOGq_up - 1, N> ct_1_d;
	R_Q_square<LOGq_up - 1, N> ct_2_d;

	ct_1 = ct;
	ct_2 = ct;

	ct_2 -= ct;


	R_Q_square<LOGq_up - LOG_qdiv, N> ct_stc, ct_stc_up, ct_stc_down;
	R_Q_square<LOGq_up - LOG_qdiv, N> ct_par_res0_up, ct_par_res0_down, ct_par_res1_up, ct_par_res1_down;

#pragma omp parallel sections
	{
#pragma omp section
		decomp< LOGq_up, N, LOG_qdiv>(ct_1, ct_par_res0_up, ct_par_res0_down);

#pragma omp section
		decomp< LOGq_up, N, LOG_qdiv>(ct_2, ct_par_res1_up, ct_par_res1_down);
	}

	tuple_SlotToCoeff<LOGq_up - LOG_qdiv, LOGN, LOGDELTA_stc, G_S, LOG_qdiv>(ct_par_res0_up, ct_par_res0_down, ct_par_res1_up, ct_par_res1_down, s, ct_stc_up, ct_stc_down);
	recomb< LOGq_up - LOG_qdiv, N, LOG_qdiv>(ct_stc_up, ct_stc_down, ct_stc);


	R_Q_square<LOGq, N> ct_base;

	RS<LOGq_up - LOG_qdiv, LOGq, N>(ct_stc, ct_base);


	std::cout << "S2C" << std::endl;


	// ModRaise
	R_Q_square<LogQ_new, N> ct_modraise;
	R_Q_square<LogQ_new_rs, N> ct_RS;
	mod_raise<LOGq, LogQ_new, N>(ct_base, ct_modraise);

	RS<LogQ_new, LogQ_new_rs, N>(ct_modraise, ct_RS);

	std::cout << "MR" << std::endl;

	auto start_1 = std::chrono::high_resolution_clock::now();


	const int LOGQ_after_cts = LogQ_new_rs - (LOGN) / G * LOGDELTA_cts + LogDelS;
	R_Q_square<LOGQ, N> ct_cts[2];
	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];
	const int LOGQ_after_evalmod = LOGQ_after_cts - 12 * LOGDELTA_boot;
	R_Q_square<LOGQ_after_evalmod, N> ct_evalmod[2];
	R_Q_square<LOGQ_after_evalmod, N> ct_evalqI[2];
	const int LOG_par = LOGQ_after_evalmod + (LOGN) / G * LOGDELTA_ori;
	R_Q_square<LOG_par, N> ct_cts_par_before;
	R_Q_square<LOG_par, N> ct_cts_par[2];
	R_Q_square<LOGQ_after_evalmod, N> ct_par_res[2];


#pragma omp parallel sections
	{
#pragma omp section
		{

			CoeffToSlot_sw<LogQ_new_rs, LOGN, LOGDELTA_cts, G>(ct_RS, s, ct_cts);


#pragma omp parallel for
			for (int i = 0; i < 2; i++)
				RS<LOGQ, LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);

			std::cout << "C2S" << std::endl;



#pragma omp parallel for
			for (int i = 0; i < 2; i++)
				EvalMod<LOGQ_after_cts, N, LOGDELTA_boot, K>(ct_ctsrs[i], s, ct_evalmod[i]);

			std::cout << "EvalMod" << std::endl;



#pragma omp parallel for
			for (int i = 0; i < 2; i++) {
				resize(ct_ctsrs[i], ct_evalqI[i]);
				ct_evalqI[i] -= ct_evalmod[i];
			}

			std::cout << "EvalqI" << std::endl;
		}

#pragma omp section
		{

			resize(ct_modraise, ct_cts_par_before);


			CoeffToSlot<LOG_par, LOGN, LOGDELTA_ori, G>(ct_cts_par_before, s, ct_cts_par);


			for (int i = 0; i < 2; i++)
				RS<LOG_par, LOGQ_after_evalmod, N>(ct_cts_par[i], ct_par_res[i]);

			std::cout << "parC2S" << std::endl;
		}
	}


	for (int i = 0; i < 2; i++) {
		ct_par_res[i] -= ct_evalqI[i];
	}

	std::cout << "comb" << std::endl;

	auto end_1 = std::chrono::high_resolution_clock::now();
	auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1);
	std::cout << "Time taken for C2S Evalmod section: " << duration_1.count() / 1000 << " s" << std::endl;



	auto start_2 = std::chrono::high_resolution_clock::now();




	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - LOG_qdiv - (LOGN) / G_S * LOG_ql;

	R_Q_square<LOGQ_after_evalmod, N> ct_boot;

	R_Q_square<LOGQ_after_evalmod, N> ct_conj_2;

	R_Q_square<2 * LOGQ_after_evalmod, 1 << LOGN> ckey2;
	HEAAN<LOGQ_after_evalmod, N>::swkgen(s_conj, s, ckey2);

	conj(ct_par_res[1], ckey2, ct_conj_2);

	ct_par_res[0] += ct_conj_2;


	auto end_2 = std::chrono::high_resolution_clock::now();
	auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - start_2);
	std::cout << "Time taken for S2C section: " << duration_2.count() / 1000 << " s" << std::endl;



	R_Q<LOGQ_after_evalmod, N> pt_out;
	HEAAN<LOGQ_after_evalmod, N>::dec(ct_par_res[0], s, pt_out);
	decode_log(pt_out, LOGDELTA, z_out);
	print_max_error<LOGN>(z, z_out);

	R_Q<LOGQ_after_evalmod, N> e;
	resize(pt, e);
	e -= pt_out;

	double e_per_Delta_sup_norm = 0;

#pragma omp parallel for reduction(max:e_per_Delta_sup_norm)
	for (int i = 0; i < N; ++i) {
		double val = (double)e[i];
		double val_abs_double = (double)std::abs(val) / Delta;

		e_per_Delta_sup_norm = val_abs_double > e_per_Delta_sup_norm ? val_abs_double : e_per_Delta_sup_norm;
	}

	std::cout << "sup_norm(pt/Delta - pt_tilde/Delta) : " << e_per_Delta_sup_norm << std::endl;
	std::cout << "Modulus consumed : " << (LOGQ - LOGQ_after_stc) << std::endl;
}


template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S, int LogDelS, int LOGDELTA_ori>
void erpluspar_all_together_np_test()
{
	std::cout << "Measuring error on all together bootstrap w erpluspar np" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

	const int N = 1 << LOGN;
	const int LogQ_new = LOGQ + LogDelS + LOGDELTA_cts;
	const int LogQ_new_rs = LOGQ + LOGDELTA_cts;

	int s[N], s_sq[N];
	HEAAN<LOGQ, N>::keygen(H, s);

	Message<LOGN> z, z_out;
	set_test_message(z);

	R_Q<LOGq, N> pt;
	R_Q_square<LOGq, N> ct;
	encode(z, Delta, pt);
	HEAAN<LOGq, N>::enc(pt, s, ct);

	// ModRaise
	R_Q_square<LogQ_new, N> ct_modraise;
	R_Q_square<LogQ_new_rs, N> ct_RS;
	mod_raise<LOGq, LogQ_new, N>(ct, ct_modraise);

	RS<LogQ_new, LogQ_new_rs, N>(ct_modraise, ct_RS);

	std::cout << "MR" << std::endl;

	auto start_1 = std::chrono::high_resolution_clock::now();


	const int LOGQ_after_cts = LogQ_new_rs - (LOGN) / G * LOGDELTA_cts + LogDelS;
	R_Q_square<LOGQ, N> ct_cts[2];
	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];
	const int LOGQ_after_evalmod = LOGQ_after_cts - 12 * LOGDELTA_boot;
	R_Q_square<LOGQ_after_evalmod, N> ct_evalmod[2];
	R_Q_square<LOGQ_after_evalmod, N> ct_evalqI[2];
	const int LOG_par = LOGQ_after_evalmod + (LOGN) / G * LOGDELTA_ori;
	R_Q_square<LOG_par, N> ct_cts_par_before;
	R_Q_square<LOG_par, N> ct_cts_par[2];
	R_Q_square<LOGQ_after_evalmod, N> ct_par_res[2];


#pragma omp parallel sections
	{
#pragma omp section
		{

			CoeffToSlot_sw<LogQ_new_rs, LOGN, LOGDELTA_cts, G>(ct_RS, s, ct_cts);


#pragma omp parallel for
			for (int i = 0; i < 2; i++)
				RS<LOGQ, LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);

			std::cout << "C2S" << std::endl;

			

#pragma omp parallel for
			for (int i = 0; i < 2; i++)
				EvalMod<LOGQ_after_cts, N, LOGDELTA_boot, K>(ct_ctsrs[i], s, ct_evalmod[i]);

			std::cout << "EvalMod" << std::endl;

			

#pragma omp parallel for
			for (int i = 0; i < 2; i++) {
				resize(ct_ctsrs[i], ct_evalqI[i]);
				ct_evalqI[i] -= ct_evalmod[i];
			}

			std::cout << "EvalqI" << std::endl;
		}

#pragma omp section
		{
			
			resize(ct_modraise, ct_cts_par_before);

			
			CoeffToSlot<LOG_par, LOGN, LOGDELTA_ori, G>(ct_cts_par_before, s, ct_cts_par);

			
			for (int i = 0; i < 2; i++)
				RS<LOG_par, LOGQ_after_evalmod, N>(ct_cts_par[i], ct_par_res[i]);

			std::cout << "parC2S" << std::endl;
		}
	}


	for (int i = 0; i < 2; i++) {
		ct_par_res[i] -= ct_evalqI[i];
	}

	std::cout << "comb" << std::endl;

	auto end_1 = std::chrono::high_resolution_clock::now();
    	auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1);
    	std::cout << "Time taken for C2S Evalmod section: " << duration_1.count()/1000 << " s" << std::endl;



	auto start_2 = std::chrono::high_resolution_clock::now();



	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - (LOGN) / G_S * LOGDELTA_stc;
	R_Q_square<LOGQ_after_evalmod, N> ct_stc;
	R_Q_square<LOGQ_after_stc, N> ct_boot;
	SlotToCoeff<LOGQ_after_evalmod, LOGN, LOGDELTA_stc, G_S>(ct_par_res[0], ct_par_res[1], s, ct_stc);
	RS<LOGQ_after_evalmod, LOGQ_after_stc, N>(ct_stc, ct_boot);

	std::cout << "S2C" << std::endl;

	auto end_2 = std::chrono::high_resolution_clock::now();
    	auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - start_2);
    	std::cout << "Time taken for S2C section: " << duration_2.count() / 1000 << " s" << std::endl;



	R_Q<LOGQ_after_stc, N> pt_out;
	HEAAN<LOGQ_after_stc, N>::dec(ct_boot, s, pt_out);
	decode_log(pt_out, LOGDELTA, z_out);
	print_max_error<LOGN>(z, z_out);

	R_Q<LOGQ_after_stc, N> e;
	resize(pt, e);
	e -= pt_out;

	double e_per_Delta_sup_norm = 0;

#pragma omp parallel for reduction(max:e_per_Delta_sup_norm)
	for (int i = 0; i < N; ++i) {
		double val = (double)e[i];
		double val_abs_double = (double)std::abs(val) / Delta;

		e_per_Delta_sup_norm = val_abs_double > e_per_Delta_sup_norm ? val_abs_double : e_per_Delta_sup_norm;
	}

	std::cout << "sup_norm(pt/Delta - pt_tilde/Delta) : " << e_per_Delta_sup_norm << std::endl;
	std::cout << "Modulus consumed : " << (LOGQ - LOGQ_after_stc) << std::endl;
}




int main()
{
	auto start_time = std::chrono::high_resolution_clock::now();

	// logN=15
	//evalroundplus_test_S2C_first<15, 28, 30, 7, 7, 49>();
	//erpluspar_12_S2C_first<15, 29, 30, 7, 7, 10, 20, 21, 49>();
	//erpluspar_all_together_bootstrap_test_S2C_first<15, 29, 30, 7, 7, 10, 20, 21, 49>();

	// logN=16
	//evalroundplus_test_S2C_first<16, 37, 42, 4, 5, 58>();
	//erpluspar_12_S2C_first<16, 38, 42, 4, 5, 16, 26, 30, 58>();
	//erpluspar_all_together_bootstrap_test_S2C_first<16, 38, 42, 4, 5, 16, 26, 30, 58>();

	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration_all = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	std::cout << "Execution time: " << duration_all.count() / 1000 << "s" << std::endl;

}
