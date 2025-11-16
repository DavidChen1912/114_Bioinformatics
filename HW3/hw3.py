import numpy as np
from Bio import SeqIO

def load_score_matrix(score_path):
    """Load substitution matrix from txt file."""
    matrix = {}
    with open(score_path) as f:
        lines = f.readlines()

    # find header row (letters)
    for i, line in enumerate(lines):
        if line.strip().startswith("A "):
            header = line.split()
            start = i + 1
            break

    for line in lines[start:]:
        row = line.split()
        aa = row[0]
        scores = list(map(int, row[1:]))
        matrix[aa] = dict(zip(header, scores))

    return matrix


def needleman_wunsch(seq1, seq2, matrix, gap_penalty):
    """Global alignment."""
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1), dtype=int)

    # initialize borders for standard forward DP
    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + gap_penalty
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + gap_penalty

    # fill dp table
    for i in range(1, m+1):
        for j in range(1, n+1):
            match = dp[i-1][j-1] + matrix[seq1[i-1]][seq2[j-1]]
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(match, delete, insert)

    # traceback from bottom-right
    def do_traceback(preference):
        # preference: tuple/list of move names in order, each in {'diag','up','left'}
        i, j = m, n
        a1, a2 = [], []
        while i > 0 or j > 0:
            moved = False
            for move in preference:
                if move == 'diag' and i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + matrix[seq1[i-1]][seq2[j-1]]:
                    a1.append(seq1[i-1])
                    a2.append(seq2[j-1])
                    i -= 1
                    j -= 1
                    moved = True
                    break
                if move == 'up' and i > 0 and dp[i][j] == dp[i-1][j] + gap_penalty:
                    a1.append(seq1[i-1])
                    a2.append('-')
                    i -= 1
                    moved = True
                    break
                if move == 'left' and j > 0 and dp[i][j] == dp[i][j-1] + gap_penalty:
                    a1.append('-')
                    a2.append(seq2[j-1])
                    j -= 1
                    moved = True
                    break
            if not moved:
                # no valid move found (shouldn't happen) -> break to avoid infinite loop
                break
        a1.reverse()
        a2.reverse()
        return ''.join(a1), ''.join(a2)

    # try several tie-breaking orders to avoid off-by-one missing-char issues
    orders = [
        ('diag', 'up', 'left'),
        ('diag', 'left', 'up'),
        ('up', 'diag', 'left'),
        ('left', 'diag', 'up'),
    ]

    for ords in orders:
        a1, a2 = do_traceback(ords)
        # if ungapped alignments match original sequences, accept
        if a1.replace('-', '') == seq1 and a2.replace('-', '') == seq2:
            return a1, a2

    # fallback to default order (diag, up, left)
    return do_traceback(('diag', 'up', 'left'))


def smith_waterman(seq1, seq2, matrix, gap_penalty):
    """Local alignment (Smith-Waterman)."""
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1), dtype=int)

    max_score = 0
    positions = []  # store all max-score positions

    for i in range(1, m+1):
        for j in range(1, n+1):
            score_diag = dp[i-1][j-1] + matrix[seq1[i-1]][seq2[j-1]]
            score_up = dp[i-1][j] + gap_penalty
            score_left = dp[i][j-1] + gap_penalty
            dp[i][j] = max(0, score_diag, score_up, score_left)

            if dp[i][j] > max_score:
                max_score = dp[i][j]
                positions = [(i, j)]
            elif dp[i][j] == max_score:
                positions.append((i, j))

    # trace back all optimal alignments (exhaustive branching DFS)
    alignments = []

    def dfs_trace(ii, jj, cur1, cur2):
        # cur1, cur2 are lists of chars built in reverse (from end to start)
        if dp[ii][jj] == 0:
            # reached beginning of local alignment; record start indices (1-based)
            a1 = ''.join(reversed(cur1))
            a2 = ''.join(reversed(cur2))
            start_i = ii + 1
            start_j = jj + 1
            end_i = i
            end_j = j
            alignments.append((a1, a2, start_i, start_j, end_i, end_j))
            return

        # consider diagonal
        if ii > 0 and jj > 0 and dp[ii][jj] == dp[ii-1][jj-1] + matrix[seq1[ii-1]][seq2[jj-1]]:
            cur1.append(seq1[ii-1])
            cur2.append(seq2[jj-1])
            dfs_trace(ii-1, jj-1, cur1, cur2)
            cur1.pop(); cur2.pop()

        # consider up (deletion in seq2)
        if ii > 0 and dp[ii][jj] == dp[ii-1][jj] + gap_penalty:
            cur1.append(seq1[ii-1])
            cur2.append('-')
            dfs_trace(ii-1, jj, cur1, cur2)
            cur1.pop(); cur2.pop()

        # consider left (insertion in seq1)
        if jj > 0 and dp[ii][jj] == dp[ii][jj-1] + gap_penalty:
            cur1.append('-')
            cur2.append(seq2[jj-1])
            dfs_trace(ii, jj-1, cur1, cur2)
            cur1.pop(); cur2.pop()


    for (i, j) in positions:
        dfs_trace(i, j, [], [])

    # post-process: for each alignment, try to extend leftwards by including diagonal pairs
    processed = []
    for a1, a2, start_i, start_j, end_i, end_j in alignments:
        # attempt to prepend diagonal matches with non-negative score
        si, sj = start_i, start_j
        prefix1 = []
        prefix2 = []
        while si > 1 and sj > 1:
            score = matrix[seq1[si-2]][seq2[sj-2]]
            if score >= 0:
                prefix1.append(seq1[si-2])
                prefix2.append(seq2[sj-2])
                si -= 1
                sj -= 1
            else:
                break
        if prefix1:
            # prepend reversed prefix (we collected leftwards)
            a1 = ''.join(reversed(prefix1)) + a1
            a2 = ''.join(reversed(prefix2)) + a2
        processed.append((a1, a2))

    alignments = processed

    # remove duplicates
    alignments = list(set(alignments))

    if not alignments:
        return []

    # compute lengths and keep only those with maximum length
    lengths = [len(a1) for a1, a2 in alignments]
    max_len = max(lengths)
    filtered = [aln for aln in alignments if len(aln[0]) == max_len]

    # sort lexicographically by protein1 then protein2
    filtered.sort(key=lambda x: (x[0], x[1]))
    return filtered


def alignment(input_path, score_path, output_path, aln, gap):
    """Main alignment function."""
    seqs = list(SeqIO.parse(input_path, "fasta"))
    seq1 = str(seqs[0].seq)
    seq2 = str(seqs[1].seq)
    id1 = seqs[0].id
    id2 = seqs[1].id

    matrix = load_score_matrix(score_path)

    if aln == "global":
        a1, a2 = needleman_wunsch(seq1, seq2, matrix, gap)
        with open(output_path, "w") as f:
            f.write(f">{id1}\n{a1}\n")
            f.write(f">{id2}\n{a2}\n")

    elif aln == "local":
        alignments = smith_waterman(seq1, seq2, matrix, gap)
        with open(output_path, "w") as f:
            for a1, a2 in alignments:
                f.write(f">{id1}\n{a1}\n")
                f.write(f">{id2}\n{a2}\n")

# ===========================================================
# 測試用（可留可刪）
# ===========================================================
if __name__ == "__main__":
    alignment("examples/test_global.fasta", "examples/pam250.txt", "examples/my_global.fasta", "global", -10)
    alignment("examples/test_local.fasta", "examples/pam100.txt", "examples/my_local.fasta", "local", -10)

