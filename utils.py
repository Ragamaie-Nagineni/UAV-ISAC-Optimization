import numpy as np

# ============================================================
#  Physical constants & channel parameters (Table II of paper)
# ============================================================
fc    = 3.5e9          # carrier frequency (Hz)
c     = 3e8            # speed of light  (m/s)
lam   = c / fc        # wavelength

Gt    = 10**(17/10)    # UAV transmitter antenna gain (17 dBi)
Gr    = 10**(17/10)    # UAV radar receiver antenna gain (17 dBi)
Gc    = 10**(0/10)     # communication receiver antenna gain (0 dBi)
sigma = 1.0            # radar cross-section (m^2)
B     = 50e6           # bandwidth (Hz)
N0    = 10**(-160/10) * 1e-3   # noise PSD (W/Hz) from -160 dBm/Hz

beta_com = (Gt * Gc * lam**2) / ((4 * np.pi)**2)
beta_rad = (Gt * Gr * lam**2 * sigma) / ((4 * np.pi)**3)


def h_com(d):
    """Communication channel power gain (LoS, eq. 12)."""
    return beta_com / (d**2 + 1e-12)


def h_rad(d):
    """Radar channel power gain (two-way path loss, eq. 13)."""
    return beta_rad / (d**4 + 1e-12)


def sinr_com(Pcom, Prad, hk_com):
    """SINR on communication channel (eq. 16)."""
    return Pcom * hk_com / (Prad * hk_com + N0 * B + 1e-30)


def sinr_rad(Prad, Pcom, hk_rad):
    """SINR on radar channel (eq. 17)."""
    return Prad * hk_rad / (Pcom * hk_rad + N0 * B + 1e-30)


def radar_rate(sinr):
    """Radar estimation rate (eq. 20): log2(1 + SINR_rad)."""
    return np.log2(1.0 + sinr)


def comm_rate(sinr):
    """Communication rate (eq. 19): log2(1 + SINR_com)."""
    return np.log2(1.0 + sinr)


def upload_rate(Pt, hc):
    """Data upload rate to collection centre (eq. 18)."""
    return np.log2(1.0 + Pt * hc / (N0 * B + 1e-30))


def distance_3d(p1, p2):
    """Euclidean distance between two 3-D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))
