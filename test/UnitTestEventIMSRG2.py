#!/usr/bin/env python3

import pyIMSRG


IMSRG2_TERMS = [
    "comm110ss",
    "comm220ss",
    "comm111ss",
    "comm121ss",
    "comm221ss",
    "comm122ss",
    "comm222_pp_hhss",
    "comm222_phss",
]


def set_terms(active_terms):
    active_terms = set(active_terms)
    for term in IMSRG2_TERMS:
        pyIMSRG.Commutator.TurnOffTerm(term)
    for term in active_terms:
        pyIMSRG.Commutator.TurnOnTerm(term)


def diff_summary(lhs, rhs):
    delta = lhs - rhs
    return abs(delta.ZeroBody), delta.OneBodyNorm(), delta.TwoBodyNorm(), delta.Norm()


def compare_case(emax, reference, terms, tolerance):
    ms = pyIMSRG.ModelSpace(emax, reference, reference)
    ut = pyIMSRG.UnitTest(ms)
    x_op = ut.RandomOp(ms, 0, 0, 0, 2, -1)
    y_op = ut.RandomOp(ms, 0, 0, 0, 2, +1)

    pyIMSRG.Commutator.SetUseIMSRG3(False)
    set_terms(terms)

    pyIMSRG.Commutator.SetIMSRG2CommutatorBackend("matrix")
    z_matrix = pyIMSRG.Commutator.CommutatorScalarScalar(x_op, y_op)

    pyIMSRG.Commutator.SetIMSRG2CommutatorBackend("event")
    z_event = pyIMSRG.Commutator.CommutatorScalarScalar(x_op, y_op)

    zero, one, two, total = diff_summary(z_event, z_matrix)
    label = "+".join(terms)
    print(
        f"event vs matrix {reference} emax={emax} terms={label}: "
        f"zero={zero:.6e} one={one:.6e} two={two:.6e} total={total:.6e}"
    )
    return max(zero, one, two, total) < tolerance


def main():
    passed = True

    passed &= compare_case(1, "He4", IMSRG2_TERMS, 1e-10)
    passed &= compare_case(2, "He6", IMSRG2_TERMS, 1e-9)

    for term in IMSRG2_TERMS:
        passed &= compare_case(2, "He6", [term], 1e-9)

    set_terms(IMSRG2_TERMS)
    pyIMSRG.Commutator.SetIMSRG2CommutatorBackend("matrix")
    print("passed? ", passed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
