import numpy as np

def generate_pam(x, input_path, output_path):
    # Dayhoff normalized amino-acid frequencies
    dayhoff_freq = {
        "A": 0.087, "R": 0.041, "N": 0.040, "D": 0.047, "C": 0.033,
        "Q": 0.038, "E": 0.050, "G": 0.089, "H": 0.034, "I": 0.037,
        "L": 0.085, "K": 0.081, "M": 0.015, "F": 0.040, "P": 0.051,
        "S": 0.070, "T": 0.058, "W": 0.010, "Y": 0.030, "V": 0.065
    }

    # 讀檔
    rows = []
    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())

    aas = rows[0]
    n = len(aas)
    f = np.array([dayhoff_freq[a] for a in aas], dtype=float)

    # 直接除以10000
    counts = np.array([[int(v) for v in r[1:1+n]] for r in rows[1:1+n]], dtype=float)
    M = counts / 10000.0

    # M^x
    Mx = np.linalg.matrix_power(M, x)

    # relatedness odds
    R = Mx / f[:, None]

    # log-odds
    eps = 1e-300
    S = 10.0 * np.log10(np.maximum(R, eps))

    # 四捨五入
    S = np.where(S >= 0, np.floor(S + 0.5), -np.floor(-S + 0.5)).astype(int)

    # 輸出
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(" " + " ".join(aas) + "\n")
        for i, aa in enumerate(aas):
            out.write(aa + " " + " ".join(f"{S[i, j]:2d}" for j in range(n)) + "\n")

# generate_pam(250, './example/mut.txt', './example/pam250_test.txt')