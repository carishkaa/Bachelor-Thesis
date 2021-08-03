from enum import Enum


class Distribution(str, Enum):
    POISSON = 'POISSON'
    NEGATIVE_BINOMIAL = 'NEGATIVE_BINOMIAL'


class PatientType(str, Enum):
    CR = 'CR'
    OT = 'OT'
    STA = 'STA'
    OTHER = 'OTHER'
