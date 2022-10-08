import re


def columns_from_formula(formula):
    if formula:
        return set(re.split("\*|\+", formula.replace(" ", "")))
    else:
        return None
