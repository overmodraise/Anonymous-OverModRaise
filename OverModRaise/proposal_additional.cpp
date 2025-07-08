#include "HEAAN/bootstrap.h"
#include "setup/rns.h"

#include <iostream>

template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int LOG_qdiv, int LOG_ql>
void S2C_3_5_bootstrap_test()
{
	std::cout << "Measuring error on S2C_3_5_bootstrap" << std::endl;
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

	// EvalqI
	R_Q_square<LOGQ_after_evalmod, N> ct_evalqI[2];
	for (int i = 0; i < 2; i++) {
		resize(ct_ctsrs[i], ct_evalqI[i]);
		ct_evalqI[i] -= ct_evalmod[i];
	}

	std::cout << "EvalqI" << std::endl;

	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - LOG_qdiv - (LOGN) / G * LOG_ql;
	R_Q_square<LOGQ_after_evalmod - LOG_qdiv, N> ct_stc, ct_stc_up, ct_stc_down;
	R_Q_square<LOGQ_after_stc, N> ct_qI;
	R_Q_square<LOGQ_after_evalmod - LOG_qdiv, N> ct_evalqI0_up, ct_evalqI0_down, ct_evalqI1_up, ct_evalqI1_down;
	decomp< LOGQ_after_evalmod, N, LOG_qdiv>(ct_evalqI[0], ct_evalqI0_up, ct_evalqI0_down);
	decomp< LOGQ_after_evalmod, N, LOG_qdiv>(ct_evalqI[1], ct_evalqI1_up, ct_evalqI1_down);

	tuple_SlotToCoeff<LOGQ_after_evalmod - LOG_qdiv, LOGN, LOGDELTA_stc, G, LOG_qdiv>(ct_evalqI0_up, ct_evalqI0_down, ct_evalqI1_up, ct_evalqI1_down, s, ct_stc_up, ct_stc_down);
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

template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int LOG_qdiv, int LOG_ql, int LogDelS>
void S2C_4_5_bootstrap_test()
{
	std::cout << "Measuring error on S2C_4_5_bootstrap" << std::endl;
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
	const int LOGQ_after_stc = LOGQ_after_evalmod - LOG_qdiv - (LOGN) / G * LOG_ql;
	R_Q_square<LOGQ_after_evalmod - LOG_qdiv, N> ct_stc, ct_stc_up, ct_stc_down;
	R_Q_square<LOGQ_after_stc, N> ct_qI;
	R_Q_square<LOGQ_after_evalmod - LOG_qdiv, N> ct_evalqI0_up, ct_evalqI0_down, ct_evalqI1_up, ct_evalqI1_down;
	decomp< LOGQ_after_evalmod, N, LOG_qdiv>(ct_evalqI[0], ct_evalqI0_up, ct_evalqI0_down);
	decomp< LOGQ_after_evalmod, N, LOG_qdiv>(ct_evalqI[1], ct_evalqI1_up, ct_evalqI1_down);

	tuple_SlotToCoeff<LOGQ_after_evalmod - LOG_qdiv, LOGN, LOGDELTA_stc, G, LOG_qdiv>(ct_evalqI0_up, ct_evalqI0_down, ct_evalqI1_up, ct_evalqI1_down, s, ct_stc_up, ct_stc_down);
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

template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S, int LOG_qdiv_C, int LOG_ql_C, int LOG_qdiv, int LOG_ql, int LogDelS>
void all_together_C2S_tuple_bootstrap_test()
{
	std::cout << "Measuring error on all_together_C2S_tuple_bootstrap" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

	const int N = 1 << LOGN;
	//const int LogDelS = 30;
	const int LogQ_new = LOGQ + LogDelS + LOG_ql_C;
	const int LogQ_new_rs = LOGQ + LOG_ql_C;

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
	const int LOGQ_after_cts = LogQ_new_rs - LOG_qdiv_C - (LOGN) / G * LOG_ql_C + LogDelS;

	R_Q_square<LogQ_new_rs - LOG_qdiv_C - LOG_ql_C, N> ct_cts[2];
	R_Q_square<LogQ_new_rs - LOG_qdiv_C, N> ctstemp_up, ctstemp_down;

	decomp< LogQ_new_rs, N, LOG_qdiv_C>(ct_RS, ctstemp_up, ctstemp_down);

	tuple_CoeffToSlot_sw<LogQ_new_rs - LOG_qdiv_C, LOGN, LOGDELTA_cts, G, LOG_qdiv_C, LOG_ql_C>(ctstemp_up, ctstemp_down, s, ct_cts);

	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];
	for (int i = 0; i < 2; i++)
		RS<LogQ_new_rs - LOG_qdiv_C - LOG_ql_C, LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);


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

template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S, int LOG_qdiv_C, int LOG_ql_C, int LOG_qdiv, int LOG_ql>
void C2S_3_5_bootstrap_test()
{
	std::cout << "Measuring error on C2S_3_5_bootstrap" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

	const int N = 1 << LOGN;
	//const int LogDelS = 30;
	const int LogQ_new = LOGQ + LOG_ql_C;

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
	const int LOGQ_after_cts = LogQ_new - LOG_qdiv_C - (LOGN) / G * LOG_ql_C ;

	R_Q_square<LogQ_new - LOG_qdiv_C - LOG_ql_C, N> ct_cts[2];
	R_Q_square<LogQ_new - LOG_qdiv_C, N> ctstemp_up, ctstemp_down;

	decomp< LogQ_new, N, LOG_qdiv_C>(ct_modraise, ctstemp_up, ctstemp_down);

	tuple_CoeffToSlot_sw<LogQ_new - LOG_qdiv_C, LOGN, LOGDELTA_cts, G, LOG_qdiv_C, LOG_ql_C>(ctstemp_up, ctstemp_down, s, ct_cts);

	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];
	for (int i = 0; i < 2; i++)
		RS<LogQ_new - LOG_qdiv_C - LOG_ql_C, LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);


	std::cout << "C2S" << std::endl;

	// EvalMod
	const int LOGQ_after_evalmod = LOGQ_after_cts - 12 * LOGDELTA_boot;
	R_Q_square<LOGQ_after_evalmod, N> ct_evalmod[2];
	for (int i = 0; i < 2; i++)
		EvalMod<LOGQ_after_cts, N, LOGDELTA_boot, K>(ct_ctsrs[i], s, ct_evalmod[i]);

	std::cout << "EvalMod" << std::endl;


	// SlotToCoeff
	const int LOGQ_after_stc = LOGQ_after_evalmod - LOG_qdiv - (LOGN) / G_S * LOG_ql;
	R_Q_square<LOGQ_after_evalmod - LOG_qdiv, N> ct_stc, ct_stc_up, ct_stc_down;
	R_Q_square<LOGQ_after_stc, N> ct_boot;
	R_Q_square<LOGQ_after_evalmod - LOG_qdiv, N> ct_evalmod0_up, ct_evalmod0_down, ct_evalmod1_up, ct_evalmod1_down;
	decomp< LOGQ_after_evalmod, N, LOG_qdiv>(ct_evalmod[0], ct_evalmod0_up, ct_evalmod0_down);
	decomp< LOGQ_after_evalmod, N, LOG_qdiv>(ct_evalmod[1], ct_evalmod1_up, ct_evalmod1_down);

	tuple_SlotToCoeff<LOGQ_after_evalmod - LOG_qdiv, LOGN, LOGDELTA_stc, G_S, LOG_qdiv>(ct_evalmod0_up, ct_evalmod0_down, ct_evalmod1_up, ct_evalmod1_down, s, ct_stc_up, ct_stc_down);
	recomb< LOGQ_after_evalmod - LOG_qdiv, N, LOG_qdiv>(ct_stc_up, ct_stc_down, ct_stc);

	RS<LOGQ_after_evalmod - LOG_qdiv, LOGQ_after_stc, N>(ct_stc, ct_boot);


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

template<int LOGN, int LOGDELTA_cts, int LOGDELTA_stc, int G, int G_S, int LOG_qdiv_C, int LOG_ql_C, int LOG_qdiv, int LOG_ql, int LogDelS>
void C2S_4_5_bootstrap_test()
{
	std::cout << "Measuring error on C2S_4_5_bootstrap" << std::endl;
	std::cout << "LOGN : " << LOGN << std::endl;
	std::cout << "LOGDELTA_cts : " << LOGDELTA_cts << std::endl;
	std::cout << "LOGDELTA_stc : " << LOGDELTA_stc << std::endl;

	const int N = 1 << LOGN;
	//const int LogDelS = 30;
	const int LogQ_new = LOGQ + LogDelS ;
	const int LogQ_new_rs = LOGQ ;

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
	const int LOGQ_after_cts = LogQ_new_rs - LOG_qdiv_C - (LOGN) / G * LOG_ql_C + LogDelS;

	R_Q_square<LogQ_new_rs - LOG_qdiv_C, N> ctstemp_up, ctstemp_down, ct_cts[2];

	decomp< LogQ_new_rs, N, LOG_qdiv_C>(ct_RS, ctstemp_up, ctstemp_down);

	tuple_CoeffToSlot<LogQ_new_rs - LOG_qdiv_C, LOGN, LOGDELTA_cts, G, LOG_qdiv_C>(ctstemp_up, ctstemp_down, s, ct_cts);

	R_Q_square<LOGQ_after_cts, N> ct_ctsrs[2];
	for (int i = 0; i < 2; i++)
		RS<LogQ_new_rs - LOG_qdiv_C, LOGQ_after_cts, N>(ct_cts[i], ct_ctsrs[i]);


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



int main()
{
	//C2S_3_5_bootstrap_test<16, 60, 50, 4, 5, 30, 30, 25, 25>();

	//C2S_4_5_bootstrap_test<16, 31, 56, 4, 5, 10, 21, 23, 33, 32>();

	//all_together_C2S_tuple_bootstrap_test<16, 31, 56, 4, 5, 10, 21, 23, 33, 32>();
}
