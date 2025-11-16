def calculate_SoP(input_path, score_path, gopen, gextend):
    import pandas as pd

    # === 1. 讀取 PAM 矩陣 ===
    with open(score_path) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    header = lines[0].split()
    matrix_data = [line.split() for line in lines[1:]]
    pam_df = pd.DataFrame(matrix_data, columns=["AA"] + header)
    pam_df.set_index("AA", inplace=True)
    pam_df = pam_df.astype(float)

    # === 2. 讀取 FASTA MSA ===
    seqs = []
    with open(input_path) as f:
        seq = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    seqs.append(seq)
                    seq = ""
            else:
                seq += line
        if seq:
            seqs.append(seq)

    n = len(seqs)
    L = len(seqs[0])
    total_score = 0

    # 紀錄前一欄是否是 gap，用來判斷 open / extend
    prev_gap = [False] * n

    # === 3. 主迴圈 ===
    for pos in range(L):
        column = [seq[pos] for seq in seqs]

        for i in range(n):
            for j in range(i + 1, n):
                a, b = column[i], column[j]

                # 若至少有一方是 gap（包含 gap–gap）
                if a == "-" or b == "-":
                    # 判斷是否延伸
                    is_extend_a = (a == "-" and prev_gap[i])
                    is_extend_b = (b == "-" and prev_gap[j])
                    is_extend = is_extend_a or is_extend_b

                    total_score += gextend if is_extend else gopen

                # 若兩邊都是字母
                else:
                    total_score += pam_df.loc[a, b]

        # 更新每條序列的 gap 狀態（for 下一欄）
        for k in range(n):
            prev_gap[k] = (column[k] == "-")

    print("SoP Score =", int(total_score))
    return int(total_score)



# === Example test ===
#if __name__ == "__main__":
    #calculate_SoP("./examples/test1.fasta", "./examples/pam250.txt", -10, -2)  # expect 1047
    #calculate_SoP("./examples/test2.fasta", "./examples/pam100.txt", -8, -2)   # expect 606
