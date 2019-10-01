# DNA Replication Problem
import time


def complementary_strand(DNA):
    bp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    output = ''
    for i in DNA:
        if i in bp.keys():
            output += bp[i]
    print(DNA)
    print(" ".join(output))


def protein_sequence(DNA):
    rs = DNA.replace(" ", "")
    DNA_split = [rs[i: i + 3] for i in range(0, len(rs), 3)]
    codons = {
        'Ala': ['GCT', 'GCC', 'GCA', 'GCG'],
        'Arg': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'Asn': ['AAT', 'AAC'],
        'Asp': ['GAT', 'GAC'],
        'Cys': ['TGT', 'TGC'],
        'Gln': ['CAA', 'CAG'],
        'Glu': ['GAA', 'GAG'],
        'Gly': ['GGT', 'GGC', 'GGA', 'GGG'],
        'His': ['CAT', 'CAC'],
        'Ile': ['ATT', 'ATC', 'ATA'],
        'Leu': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
        'Lys': ['AAA', 'AAG'],
        'Met': ['ATG'],
        'Phe': ['TTT', 'TTC'],
        'Pro': ['CCT', 'CCC', 'CCA', 'CCG'],
        'Ser': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
        'Thr': ['ACT', 'ACC', 'ACA', 'ACG'],
        'Trp': ['TGG'],
        'Tyr': ['TAT', 'TAC'],
        'Val': ['GTT', 'GTC', 'GTA', 'GTG'],
        'STOP': ['TAA', 'TGA', 'TAG']
    }
    cdns = {i: k for k, v in codons.items() for i in v}
    result = [cdns[j] for j in DNA_split]
    print(DNA)
    print(' '.join(result))


t0 = time.clock()
complementary_strand("A A T G C C T A T G G C")
t1 = time.clock()
print("{0} ms\n".format((t1 - t0) * 1000))
t2 = time.clock()
protein_sequence("A T G T T T C G A G G C T A A")
t3 = time.clock()
print("{0} ms".format((t3 - t2) * 1000))
