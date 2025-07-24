import os
import math
import random
import numpy as np
from pyrosetta import rosetta
from pyrosetta import init
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta.rosetta.core.pack.task.operation import IncludeCurrent
from pyrosetta.rosetta.core.pack.task import operation as task_op
from pyrosetta.rosetta.core.scoring import ScoreFunctionFactory
from pyrosetta.rosetta.core.scoring import score_type_from_name
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
from pyrosetta.rosetta.core.scoring.func import CircularHarmonicFunc
from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
from pyrosetta.rosetta.core.scoring.constraints import DihedralConstraint
from pyrosetta.rosetta.core.scoring.constraints import AngleConstraint
from pyrosetta.rosetta.core.pose import addVirtualResAsRoot
from pyrosetta.rosetta.core.pose import pdbslice
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
from pyrosetta.rosetta.utility import vector1_unsigned_long

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

print(" ")
print("Initializing PyRosetta...")

print(" ")
init("-ignore_unrecognized_res -ex1 -ex2 -ex2aro -ex3 -ex4 -extrachi_cutoff 0 -use_input_sc -detect_disulf -no_optH false -flip_HNQ -mute core.scoring.etable basic.io.database core.chemical.GlobalResidueTypeSet core.import_pose.import_pose core.io.pdb.file_data core.io.pose_from_sfr.PoseFromSFRBuilder core.io.pose_from_sfr.chirality_resolution core.energy_methods.CartesianBondedEnergy")

print(" ")
print("PyRosetta initialized.")

def windows_to_wsl_path(path):
    path = path.strip('"\'')
    if len(path) >= 3 and path[1] == ':' and path[2] == '\\':
        drive = path[0].lower()
        path = f"/mnt/{drive}" + path[2:]
        path = path.replace('\\', '/')
    return path

input_pdb  = windows_to_wsl_path(r"C:\Users\Simon-Alexandre\Documents\Doctorat\Structures\3_St1Cas9\2_St1Cas9_DGCC7710_AcrIIA6\run_2\St1Cas9_DGCC7710_RNA_AcrIIA6_relaxed.pdb")

print(" ")
pose = pose_from_pdb(input_pdb)
print(" ")
print(f"Loaded pose from: {input_pdb}")

addVirtualResAsRoot(pose)
anchor_res = pose.total_residue()
anchor_atom_id = AtomID(1, anchor_res)

scorefxn_cart = ScoreFunctionFactory.create_score_function("ref2015_cart")
fa_atr_init                = 1.0
fa_rep_init                = 0.55
fa_sol_init                = 1.0
fa_intra_rep_init          = 0.005
fa_intra_sol_xover4_init   = 1.0
lk_ball_wtd_init           = 1.0
fa_elec_init               = 1.0
hbond_sr_bb_init           = 1.0
hbond_lr_bb_init           = 1.0
hbond_bb_sc_init           = 1.0
hbond_sc_init              = 1.0
dslf_fa13_init             = 1.25
omega_init                 = 0.6
fa_dun_init                = 0.7
p_aa_pp_init               = 0.6
yhh_planarity_init         = 0.625
ref_init                   = 1.0
rama_prepro_init           = 0.7
cart_bonded_init           = 0.8
rna_torsion_init           = 1.5
rna_sugar_close_init       = 1.0
dna_bb_torsion_init        = 1.5
dna_sugar_close_init       = 1.0
fa_stack_init              = 1.0
dihedral_init              = 1.0
atom_pair_init             = 1.0
print(" ")
print("Score function configured")

tf_base = TaskFactory()
tf_base.push_back(RestrictToRepacking())
tf_base.push_back(IncludeCurrent())
packer_task_base = tf_base.create_task_and_apply_taskoperations(pose)
print(" ")
print("TaskFactory configured.")

n_chains = pose.num_chains()
chain_starts = [pose.chain_begin(i+1) for i in range(n_chains)]
chain_ends = [pose.chain_end(i+1) for i in range(n_chains)]
virtual_root = pose.total_residue() 

print(" ")
print(f"Chains detected: {n_chains}")
for i in range(n_chains):
    chain_id = pose.pdb_info().chain(chain_starts[i])
    print(f"Chain {chain_id}: {chain_starts[i]}-{chain_ends[i]}")

print(" ")
print(f"Virtual root residue: {virtual_root}")

ft = rosetta.core.kinematics.FoldTree()
for i in range(n_chains):
    ft.add_edge(virtual_root, chain_starts[i], i+1)
for i in range(n_chains):
    ft.add_edge(chain_starts[i], chain_ends[i], -1)

if ft.check_fold_tree():
    pose.fold_tree(ft)
    print(" ")
    print("Fold tree configured.")
else:
    print("Fold tree invalid!")

print(" ")

print("Jump setup:")
for i in range(n_chains):
    chain_id = pose.pdb_info().chain(chain_starts[i])
    print(f"Jump {i+1}: virtual root -> chain {chain_id} ({chain_starts[i]})")

input_dir = os.path.dirname(input_pdb)
input_base = os.path.splitext(os.path.basename(input_pdb))[0]
run_number = 1
while True:
    output_dir = os.path.join(input_dir, f"output_run_{run_number}")
    try:
        os.makedirs(output_dir)
        break
    except FileExistsError:
        run_number += 1

log_path = os.path.join(output_dir, f"run_log_{run_number}.txt")  
log_file = open(log_path, "w")
def log_print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file)

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# --- (A) CONSTRAINT PARAMETERS: control the strength and type of structural restraints --- #

interface_cutoff = 5.0                     # (3) Distance cutoff for interface residue detection (Å)
interface_distance_stddev = 0.25           # (4) Stddev for interface distance constraints (Å)
interface_backbone_distance_cutoff = 10.0  # (5) Max distance for backbone interface constraints (Å)
wc_distance_cutoff = 3.5                   # (6) Distance cutoff for identifying Watson-Crick base pairs (Å)

prot_angle_stddev = 1.5                    # (10) Stddev for protein backbone angle constraints (degrees)
prot_backbone_stddev_deg = 15.0            # (11) Stddev for protein backbone dihedral constraints (degrees)
prot_planarity_stddev_deg = 5.0            # (12) Stddev for aromatic ring planarity in proteins (degrees)
prot_c_n_stddev = 1.0                      # (13) Stddev for C-N peptide bond length in proteins (Å)

nuc_bond_stddev = 0.015                    # (14) Stddev for nucleic acid bond length constraints (Å)
nuc_angle_stddev = 2.0                     # (15) Stddev for nucleic acid bond angle constraints (degrees)
nuc_pucker_stddev_deg = 10.0               # (16) Stddev for sugar pucker dihedral in nucleic acids (degrees)
nuc_hbond_stddev = 1.0                     # (17) Stddev for Watson-Crick H-bond constraints (Å)
nuc_coplanarity_stddev_deg = 2.0           # (18) Stddev for base pair coplanarity (degrees)
nuc_backbone_stddev_deg = 15.0             # (19) Stddev for nucleic acid backbone dihedrals (degrees)
nuc_planarity_stddev_deg = 5.0             # (20) Stddev for nucleic acid base planarity (degrees)
nuc_critical_angle_stddev = 1.5            # (21) Stddev for critical backbone angles (degrees)
nuc_all_angle_stddev = 1.0                 # (22) Stddev for all nucleic acid angles (degrees)
nuc_o3p_p_stddev = 0.06                    # (23) Stddev for O3'-P bond (Å)

# --- (B) INTERFACE SAMPLING PARAMETERS: control the interface sampling process --- #

interface1_name = ""
interface1_jump_chain = ""
interface1_group_a = ['', '']
interface1_group_b = ['', '']

interface2_name = ""
interface2_jump_chain = ""
interface2_group_a = ['']
interface2_group_b = ['']

interface_config = {
    interface1_name : {"jump_chain": interface1_jump_chain, "group_a": interface1_group_a, "group_b": interface1_group_b},
    interface2_name : {"jump_chain": interface2_jump_chain, "group_a": interface2_group_a, "group_b": interface2_group_b}
}

bsa_contact_cutoff = 5.0                  # (1) Distance cutoff for BSA contact (Å)
bsa_per_contact = 30.0                    # (2) Estimated BSA per contact (Å²)

rb_translation_mag = 0.2                  # (3) Magnitude for random translation (Å)
rb_rotation_mag_deg = 0.5                 # (4) Magnitude for random rotation (degrees)

interface_separation_distance = 500.0     # (5) Distance for separating chains in ΔΔG (Å)

num_clones = 10                           # (6) Number of interface sampling clones

min_max_iter = 100                        # (7) Maximum minimization iterations
min_tolerance = 0.1                       # (8) Minimization tolerance


def pylist_to_vector1_unsigned_long(pylist):

    v = vector1_unsigned_long(len(pylist))
    for i, val in enumerate(pylist):
        v[i+1] = val 

    return v

def extract_subpose_by_chains(pose, chain_list):
    """Returns a subpose containing only the specified chains."""
    pdb_info = pose.pdb_info()
    res_indices = []
    for i in range(1, pose.total_residue() + 1):
        if pdb_info.chain(i) in chain_list:
            res_indices.append(i)
    subpose = rosetta.core.pose.Pose()
    pdbslice(subpose, pose, pylist_to_vector1_unsigned_long(res_indices))
    return subpose

def residue_composition(residue_indices, pose):
    hydrophobic = set('AVLIMFWY')
    polar = set('STNQCGP')
    charged = set('DEKRH')
    comp = {'hydrophobic': 0, 'polar': 0, 'charged': 0}
    for idx in residue_indices:
        res = pose.residue(idx)
        aa = res.name1()
        if aa in hydrophobic:
            comp['hydrophobic'] += 1
        elif aa in polar:
            comp['polar'] += 1
        elif aa in charged:
            comp['charged'] += 1
    return comp

def get_interface_residues(pose, chain_indices, cutoff=None):
    """ Returns a set of residue indices in the given chains that are within cutoff Å of any residue in another chain. """
    if cutoff is None:
        cutoff = interface_cutoff
    interface_res = set()
    nres = pose.total_residue()
    chainA = pose.pdb_info().chain(chain_indices[0])
    chainB = pose.pdb_info().chain(chain_indices[1])
    groupA = [idx for idx in range(1, nres + 1) if pose.pdb_info().chain(idx) == chainA]
    groupB = [idx for idx in range(1, nres + 1) if pose.pdb_info().chain(idx) == chainB]
    for res1 in groupA:
        r1 = pose.residue(res1)
        for res2 in groupB:
            r2 = pose.residue(res2)
            for a1 in range(1, r1.natoms() + 1):
                if r1.atom_name(a1).strip().startswith('H'):
                    continue
                xyz1 = r1.xyz(a1)
                for a2 in range(1, r2.natoms() + 1):
                    if r2.atom_name(a2).strip().startswith('H'):
                        continue
                    xyz2 = r2.xyz(a2)
                    if (xyz1 - xyz2).norm() < cutoff:
                        interface_res.add(res1)
                        interface_res.add(res2)
                        break
                else:
                    continue
                break
    return interface_res

def add_interface_constraints(pose, interface_residues_dict, distance_stddev=None, backbone_distance_cutoff=None):
    """ Universal interface constraint: CA for proteins, C4' for nucleic acids (else P). """
    if distance_stddev is None:
        distance_stddev = interface_distance_stddev
    if backbone_distance_cutoff is None:
        backbone_distance_cutoff = interface_backbone_distance_cutoff
    def main_atom(res):
        if res.is_protein() and res.has("CA"):
            return "CA"
        elif (res.is_DNA() or res.is_RNA()):
            if res.has("C4'"):
                return "C4'"
            elif res.has("P"):
                return "P"
        return None
    n_constraints = 0
    for key, value in interface_residues_dict.items():
        residues = value["residues"]
        chainA, chainB = value["chains"]
        groupA = [idx for idx in residues if pose.pdb_info().chain(idx) == chainA]
        groupB = [idx for idx in residues if pose.pdb_info().chain(idx) == chainB]
        for res1_idx in groupA:
            res1 = pose.residue(res1_idx)
            atom1 = main_atom(res1)
            if not atom1:
                continue
            for res2_idx in groupB:
                res2 = pose.residue(res2_idx)
                atom2 = main_atom(res2)
                if not atom2:
                    continue
                xyz1 = res1.xyz(atom1)
                xyz2 = res2.xyz(atom2)
                dist = (xyz1 - xyz2).norm()
                if dist < backbone_distance_cutoff:
                    id1 = AtomID(res1.atom_index(atom1), res1_idx)
                    id2 = AtomID(res2.atom_index(atom2), res2_idx)
                    func = HarmonicFunc(dist, distance_stddev)
                    pose.add_constraint(AtomPairConstraint(id1, id2, func))
                    n_constraints += 1
    print(f"Added {n_constraints} universal interface constraints across all chain pairs.")
    
def identify_watson_crick_pairs_by_criteria(pose, distance_cutoff=None):
    """ Identify Watson-Crick base pairs based on distance and chain criteria. """
    if distance_cutoff is None:
        distance_cutoff = wc_distance_cutoff
    wc_patterns = {
        ('A', 'T'): [('N6', 'O4'), ('N1', 'N3')], ('T', 'A'): [('N3', 'N1'), ('O4', 'N6')],
        ('G', 'C'): [('N1', 'N3'), ('N2', 'O2'), ('O6', 'N4')], ('C', 'G'): [('N3', 'N1'), ('O2', 'N2'), ('N4', 'O6')],
        ('A', 'U'): [('N6', 'O4'), ('N1', 'N3')], ('U', 'A'): [('N3', 'N1'), ('O4', 'N6')]
    }
    def get_base_type(residue):
        name = residue.name3().strip()
        base_map = {
            'DA': 'A', 'DT': 'T', 'DG': 'G', 'DC': 'C', 'A': 'A', 'T': 'T', 'G': 'G', 'C': 'C', 'U': 'U',
            'rA': 'A', 'rU': 'U', 'rG': 'G', 'rC': 'C', 'ADE': 'A', 'THY': 'T', 'GUA': 'G', 'CYT': 'C', 'URA': 'U'
        }
        return base_map.get(name, name)
    def get_chain_id(pose, res_idx):
        return pose.pdb_info().chain(res_idx) if pose.pdb_info() else 'A'
    def check_base_pair_distance(pose, res1_idx, res2_idx, atom1, atom2):
        res1 = pose.residue(res1_idx)
        res2 = pose.residue(res2_idx)
        if not (res1.has(atom1) and res2.has(atom2)):
            return False, 0.0
        xyz1 = res1.xyz(atom1)
        xyz2 = res2.xyz(atom2)
        distance = (xyz1 - xyz2).norm()
        return distance <= distance_cutoff, distance
    def meets_chain_criteria(pose, res1_idx, res2_idx, res1, res2):
        chain1 = get_chain_id(pose, res1_idx)
        chain2 = get_chain_id(pose, res2_idx)
        if chain1 != chain2:
            return True, f"inter-chain ({chain1}-{chain2})"
        if res1.is_RNA() and res2.is_RNA() and chain1 == chain2:
            return True, f"intra-RNA ({chain1})"
        if res1.is_DNA() and res2.is_DNA() and chain1 == chain2:
            return False, f"intra-DNA ({chain1}) - excluded"
        if ((res1.is_DNA() and res2.is_RNA()) or (res1.is_RNA() and res2.is_DNA())) and chain1 == chain2:
            return True, f"mixed DNA-RNA ({chain1})"
        return False, "unknown"
    def is_watson_crick_pair(pose, res1_idx, res2_idx):
        res1 = pose.residue(res1_idx)
        res2 = pose.residue(res2_idx)
        if not ((res1.is_DNA() or res1.is_RNA()) and (res2.is_DNA() or res2.is_RNA())):
            return False, None, 0.0, None
        meets_criteria, criteria_type = meets_chain_criteria(pose, res1_idx, res2_idx, res1, res2)
        if not meets_criteria:
            return False, None, 0.0, criteria_type
        base1 = get_base_type(res1)
        base2 = get_base_type(res2)
        pair_key = (base1, base2)
        if pair_key not in wc_patterns:
            return False, None, 0.0, criteria_type
        required_bonds = wc_patterns[pair_key]
        valid_bonds = 0
        min_distance = float('inf')
        for atom1, atom2 in required_bonds:
            is_valid, distance = check_base_pair_distance(pose, res1_idx, res2_idx, atom1, atom2)
            if is_valid:
                valid_bonds += 1
                min_distance = min(min_distance, distance)
        min_bonds_required = min(2, len(required_bonds))
        if valid_bonds >= min_bonds_required:
            pair_type = f"{base1}-{base2}"
            return True, pair_type, min_distance, criteria_type
        return False, None, 0.0, criteria_type
    protected_pairs = []
    nucleic_residues = []
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_DNA() or res.is_RNA():
            nucleic_residues.append(i)
    for i, res1_idx in enumerate(nucleic_residues):
        for res2_idx in nucleic_residues[i+1:]:
            is_wc, pair_type, distance, criteria_type = is_watson_crick_pair(pose, res1_idx, res2_idx)
            if is_wc:
                protected_pairs.append((res1_idx, res2_idx, pair_type, distance, criteria_type))
    return protected_pairs

def add_nucleic_acid_constraints(pose, bond_stddev=None, angle_stddev=None, pucker_stddev_deg=None, hbond_stddev=None, coplanarity_stddev_deg=None, backbone_stddev_deg=None, planarity_stddev_deg=None, critical_angle_stddev=None, all_angle_stddev=None, o3p_p_stddev=None):
    """ Comprehensive nucleic acid constraints. All stddevs are parametrized. """
    bond_stddev = bond_stddev if bond_stddev is not None else nuc_bond_stddev
    angle_stddev = angle_stddev if angle_stddev is not None else nuc_angle_stddev
    pucker_stddev_deg = pucker_stddev_deg if pucker_stddev_deg is not None else nuc_pucker_stddev_deg
    hbond_stddev = hbond_stddev if hbond_stddev is not None else nuc_hbond_stddev
    coplanarity_stddev_deg = coplanarity_stddev_deg if coplanarity_stddev_deg is not None else nuc_coplanarity_stddev_deg
    backbone_stddev_deg = backbone_stddev_deg if backbone_stddev_deg is not None else nuc_backbone_stddev_deg
    planarity_stddev_deg = planarity_stddev_deg if planarity_stddev_deg is not None else nuc_planarity_stddev_deg
    critical_angle_stddev = critical_angle_stddev if critical_angle_stddev is not None else nuc_critical_angle_stddev
    all_angle_stddev = all_angle_stddev if all_angle_stddev is not None else nuc_all_angle_stddev
    o3p_p_stddev = o3p_p_stddev if o3p_p_stddev is not None else nuc_o3p_p_stddev

    pose.update_residue_neighbors()
    pose.conformation().detect_bonds()
    pucker_stddev_rad = math.radians(pucker_stddev_deg)
    backbone_stddev_rad = math.radians(backbone_stddev_deg)
    coplanarity_stddev_rad = math.radians(coplanarity_stddev_deg)
    planarity_stddev_rad = math.radians(planarity_stddev_deg) 

    protected_pairs = identify_watson_crick_pairs_by_criteria(pose)
    wc_patterns = {
        ('A', 'T'): [('N6', 'O4'), ('N1', 'N3')], ('T', 'A'): [('N3', 'N1'), ('O4', 'N6')],
        ('G', 'C'): [('N1', 'N3'), ('N2', 'O2'), ('O6', 'N4')], ('C', 'G'): [('N3', 'N1'), ('O2', 'N2'), ('N4', 'O6')],
        ('A', 'U'): [('N6', 'O4'), ('N1', 'N3')], ('U', 'A'): [('N3', 'N1'), ('O4', 'N6')]
    }
    purine_dihedrals = [('N9', 'C8', 'N7', 'C5'), ('C4', 'C5', 'C6', 'N1'), ('C6', 'N1', 'C2', 'N3')]
    pyrimidine_dihedrals = [('N1', 'C2', 'N3', 'C4'), ('C5', 'C4', 'N3', 'C2')]
    nucleic_torsions = {
        'alpha': [(0, "O3'", "P", "O5'", "C5'")], 'beta': [(0, "P", "O5'", "C5'", "C4'")], 'gamma': [(0, "O5'", "C5'", "C4'", "C3'")],
        'delta': [(0, "C5'", "C4'", "C3'", "O3'")], 'epsilon': [(0, "C4'", "C3'", "O3'", "P")], 'zeta': [(0, "C3'", "O3'", "P", "O5'")],
        'chi_pur': [(0, "O4'", "C1'", "N9", "C4")], 'chi_pyr': [(0, "O4'", "C1'", "N1", "C2")],
        'nu0': [(0, "C4'", "O4'", "C1'", "C2'")], 'nu1': [(0, "O4'", "C1'", "C2'", "C3'")], 'nu2': [(0, "C1'", "C2'", "C3'", "C4'")],
        'nu3': [(0, "C2'", "C3'", "C4'", "O4'")], 'nu4': [(0, "C3'", "C4'", "O4'", "C1'")]
    }
    def get_base_type(residue):
        name = residue.name3().strip()
        base_map = {
            'DA': 'A', 'DT': 'T', 'DG': 'G', 'DC': 'C', 'A': 'A', 'T': 'T', 'G': 'G', 'C': 'C', 'U': 'U',
            'rA': 'A', 'rU': 'U', 'rG': 'G', 'rC': 'C', 'ADE': 'A', 'THY': 'T', 'GUA': 'G', 'CYT': 'C', 'URA': 'U'
        }
        return base_map.get(name, name)
    def base_atoms(res):
        return ("N9", "C8") if res.is_purine() else ("N1", "C6")
    hbond_count = 0
    coplanarity_count = 0

    for res1_idx, res2_idx, pair_type, distance, criteria_type in protected_pairs:
        res1 = pose.residue(res1_idx)
        res2 = pose.residue(res2_idx)
        base1 = get_base_type(res1)
        base2 = get_base_type(res2)
        pair_key = (base1, base2)
        if pair_key in wc_patterns:
            for atom1_name, atom2_name in wc_patterns[pair_key]:
                if res1.has(atom1_name) and res2.has(atom2_name):
                    atom1_id = AtomID(res1.atom_index(atom1_name), res1_idx)
                    atom2_id = AtomID(res2.atom_index(atom2_name), res2_idx)
                    current_distance = (res1.xyz(atom1_name) - res2.xyz(atom2_name)).norm()
                    tight_stddev = min(hbond_stddev * 0.5, 0.1)
                    func = HarmonicFunc(current_distance, tight_stddev)
                    pose.add_constraint(AtomPairConstraint(atom1_id, atom2_id, func))
                    hbond_count += 1
        a1_i, a2_i = base_atoms(res1)
        a1_j, a2_j = base_atoms(res2)
        if res1.has(a1_i) and res1.has(a2_i) and res2.has(a1_j) and res2.has(a2_j):
            ids = [AtomID(res1.atom_index(a1_i), res1_idx), AtomID(res1.atom_index(a2_i), res1_idx),
                   AtomID(res2.atom_index(a1_j), res2_idx), AtomID(res2.atom_index(a2_j), res2_idx)]
            current_dihedral = rosetta.numeric.dihedral_degrees(res1.xyz(a1_i), res1.xyz(a2_i), res2.xyz(a1_j), res2.xyz(a2_j))
            target_angle = 0.0 if abs(current_dihedral) < 90.0 else 180.0
            target_rad = math.radians(target_angle)
            func = CircularHarmonicFunc(target_rad, coplanarity_stddev_rad)
            pose.add_constraint(DihedralConstraint(*ids, func))
            coplanarity_count += 1

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if not (res.is_RNA() or res.is_DNA()):
            continue
        ring_bonds = [("C1'", "O4'"), ("O4'", "C4'"), ("C4'", "C3'"), ("C3'", "C2'"), ("C2'", "C1'")]
        for atom1, atom2 in ring_bonds:
            if res.has(atom1) and res.has(atom2):
                id1 = AtomID(res.atom_index(atom1), i)
                id2 = AtomID(res.atom_index(atom2), i)
                dist = (res.xyz(atom1) - res.xyz(atom2)).norm()
                func = HarmonicFunc(dist, bond_stddev)
                pose.add_constraint(AtomPairConstraint(id1, id2, func))
        ring_angles = [("C1'", "O4'", "C4'"), ("O4'", "C4'", "C3'"), ("C4'", "C3'", "C2'"), ("C3'", "C2'", "C1'"), ("C2'", "C1'", "O4'")]
        for atoms in ring_angles:
            if all(res.has(a) for a in atoms):
                ids = [AtomID(res.atom_index(a), i) for a in atoms]
                current_angle = rosetta.numeric.angle_degrees(res.xyz(atoms[0]), res.xyz(atoms[1]), res.xyz(atoms[2]))
                func = HarmonicFunc(current_angle, angle_stddev)
                pose.add_constraint(AngleConstraint(*ids, func))
        ring_dihedrals = [("C4'", "O4'", "C1'", "C2'"), ("O4'", "C1'", "C2'", "C3'"), ("C1'", "C2'", "C3'", "C4'"), ("C2'", "C3'", "C4'", "O4'"), ("C3'", "C4'", "O4'", "C1'")]
        for atoms in ring_dihedrals:
            if all(res.has(a) for a in atoms):
                ids = [AtomID(res.atom_index(a), i) for a in atoms]
                current = math.radians(rosetta.numeric.dihedral_degrees(res.xyz(atoms[0]), res.xyz(atoms[1]), res.xyz(atoms[2]), res.xyz(atoms[3])))
                func = CircularHarmonicFunc(current, pucker_stddev_rad)
                pose.add_constraint(DihedralConstraint(*ids, func))
        critical_angles = [("O3'", "P", "O5'", 104.0), ("P", "O5'", "C5'", 120.0), ("O5'", "C5'", "C4'", 109.0), ("C5'", "C4'", "O4'", 109.0), ("C4'", "O4'", "C1'", 109.0)]
        for atoms in critical_angles:
            if len(atoms) == 4 and all(res.has(a) for a in atoms[:3]):
                ids = [AtomID(res.atom_index(a), i) for a in atoms[:3]]
                func = HarmonicFunc(atoms[3], critical_angle_stddev )
                pose.add_constraint(AngleConstraint(*ids, func))
        all_nucleic_angles = [("C1'", "O4'", "C4'"), ("O4'", "C4'", "C3'"), ("C4'", "C3'", "C2'"), ("C3'", "C2'", "C1'"), ("C2'", "C1'", "O4'"), ("O4'", "C1'", "C2'"), ("C1'", "C2'", "C3'"), ("C2'", "C3'", "C4'"), ("C3'", "C4'", "O4'"), ("C4'", "O4'", "C1'"), ("P", "O5'", "C5'"), ("O5'", "C5'", "C4'"), ("C5'", "C4'", "C3'"), ("C4'", "C3'", "O3'")]
        for atoms in all_nucleic_angles:
            if all(res.has(a) for a in atoms):
                ids = [AtomID(res.atom_index(a), i) for a in atoms]
                current_angle = rosetta.numeric.angle_degrees(res.xyz(atoms[0]), res.xyz(atoms[1]), res.xyz(atoms[2]))
                func = HarmonicFunc(current_angle, all_angle_stddev )
                pose.add_constraint(AngleConstraint(*ids, func))
        dihedrals = purine_dihedrals if res.is_purine() else pyrimidine_dihedrals
        for atoms in dihedrals:
            if all(res.has(atom) for atom in atoms):
                ids = [AtomID(res.atom_index(atom), i) for atom in atoms]
                xyzs = [res.xyz(atom) for atom in atoms]
                angle = rosetta.numeric.dihedral_degrees(*xyzs)
                func = CircularHarmonicFunc(math.radians(angle), planarity_stddev_rad)
                pose.add_constraint(DihedralConstraint(*ids, func))
        torsions = nucleic_torsions.copy()
        if res.is_purine():
            torsions['chi'] = torsions['chi_pur']
        else:
            torsions['chi'] = torsions['chi_pyr']
        torsions.pop('chi_pur')
        torsions.pop('chi_pyr')
        for torsion_name, torsion_list in torsions.items():
            for offset, a1, a2, a3, a4 in torsion_list:
                res_idx = i + offset
                if res_idx < 1 or res_idx > pose.total_residue():
                    continue
                target_res = pose.residue(res_idx)
                if not all(target_res.has(atom) for atom in [a1, a2, a3, a4]):
                    continue
                ids = [AtomID(target_res.atom_index(atom), res_idx) for atom in [a1, a2, a3, a4]]
                angle_deg = rosetta.numeric.dihedral_degrees(target_res.xyz(a1), target_res.xyz(a2), target_res.xyz(a3), target_res.xyz(a4))
                angle_rad = math.radians(angle_deg)
                func = CircularHarmonicFunc(angle_rad, backbone_stddev_rad)
                pose.add_constraint(DihedralConstraint(*ids, func))
    nres = pose.total_residue()
    for i in range(1, nres):
        res_i = pose.residue(i)
        res_j = pose.residue(i + 1)
        if (res_i.is_DNA() or res_i.is_RNA()) and (res_j.is_DNA() or res_j.is_RNA()):
            if res_i.has("O3'") and res_j.has("P"):
                id1 = AtomID(res_i.atom_index("O3'"), i)
                id2 = AtomID(res_j.atom_index("P"), i + 1)
                dist = (res_i.xyz("O3'") - res_j.xyz("P")).norm()
                func = HarmonicFunc(dist, o3p_p_stddev)
                pose.add_constraint(AtomPairConstraint(id1, id2, func))
    print(f"Nucleic acid constraints added: {hbond_count} H-bonds, {coplanarity_count} coplanarity for {len(protected_pairs)} base pairs.")
    criteria_counts = {}
    for _, _, _, _, criteria_type in protected_pairs:
        criteria_counts[criteria_type] = criteria_counts.get(criteria_type, 0) + 1
    print(f" ")
    print("Protected base pairs by type:")
    for criteria, count_pairs in criteria_counts.items():
        print(f"{criteria}: {count_pairs} pairs")

def add_protein_constraints(pose, angle_stddev=None, backbone_stddev_deg=None, planarity_stddev_deg=None, c_n_stddev=None):
    """ Comprehensive protein constraints. All stddevs are parametrized. """
    angle_stddev = angle_stddev if angle_stddev is not None else prot_angle_stddev
    backbone_stddev_deg = backbone_stddev_deg if backbone_stddev_deg is not None else prot_backbone_stddev_deg
    planarity_stddev_deg = planarity_stddev_deg if planarity_stddev_deg is not None else prot_planarity_stddev_deg
    c_n_stddev = c_n_stddev if c_n_stddev is not None else prot_c_n_stddev

    pose.update_residue_neighbors()
    pose.conformation().detect_bonds()
    backbone_stddev_rad = math.radians(backbone_stddev_deg) 
    planarity_stddev_rad = math.radians(planarity_stddev_deg)

    aromatic_dihedrals = {
        'PHE': [('CG', 'CD1', 'CE1', 'CZ'), ('CG', 'CD2', 'CE2', 'CZ'), ('CD1', 'CE1', 'CZ', 'CE2'), ('CD2', 'CE2', 'CZ', 'CE1')],
        'TYR': [('CG', 'CD1', 'CE1', 'CZ'), ('CG', 'CD2', 'CE2', 'CZ'), ('CD1', 'CE1', 'CZ', 'CE2'), ('CD2', 'CE2', 'CZ', 'CE1')],
        'TRP': [('CD2', 'CE2', 'NE1', 'CD1'), ('CG', 'CD1', 'NE1', 'CE2'), ('CD2', 'CE2', 'CZ2', 'CH2'), ('CE2', 'CZ2', 'CH2', 'CZ3')],
        'HIS': [('CG', 'ND1', 'CE1', 'NE2'), ('CG', 'CD2', 'NE2', 'CE1')]
    }
    nres = pose.total_residue()
    for i in range(1, nres + 1):
        res = pose.residue(i)
        if not res.is_protein():
            continue
        if res.has("N") and res.has("CA") and res.has("C"):
            id1 = AtomID(res.atom_index("N"), i)
            id2 = AtomID(res.atom_index("CA"), i)
            id3 = AtomID(res.atom_index("C"), i)
            current_angle = rosetta.numeric.angle_degrees(res.xyz("N"), res.xyz("CA"), res.xyz("C"))
            func = HarmonicFunc(current_angle, angle_stddev)
            pose.add_constraint(AngleConstraint(id1, id2, id3, func))
        resname = res.name3()
        if resname in aromatic_dihedrals and planarity_stddev_rad is not None:
            for atoms in aromatic_dihedrals[resname]:
                if all(res.has(atom) for atom in atoms):
                    ids = [AtomID(res.atom_index(atom), i) for atom in atoms]
                    xyzs = [res.xyz(atom) for atom in atoms]
                    angle = rosetta.numeric.dihedral_degrees(*xyzs)
                    func = CircularHarmonicFunc(math.radians(angle), planarity_stddev_rad)
                    pose.add_constraint(DihedralConstraint(*ids, func))
        if i > 1:
            prev_res = pose.residue(i - 1)
            if all([prev_res.has("C"), res.has("N"), res.has("CA"), res.has("C")]):
                ids = [AtomID(prev_res.atom_index("C"), i - 1), AtomID(res.atom_index("N"), i), AtomID(res.atom_index("CA"), i), AtomID(res.atom_index("C"), i)]
                angle_deg = rosetta.numeric.dihedral_degrees(prev_res.xyz("C"), res.xyz("N"), res.xyz("CA"), res.xyz("C"))
                angle_rad = math.radians(angle_deg)
                func = CircularHarmonicFunc(angle_rad, backbone_stddev_rad)
                pose.add_constraint(DihedralConstraint(*ids, func))
        if i < nres:
            next_res = pose.residue(i + 1)
            if all([res.has("N"), res.has("CA"), res.has("C"), next_res.has("N")]):
                ids = [AtomID(res.atom_index("N"), i), AtomID(res.atom_index("CA"), i), AtomID(res.atom_index("C"), i), AtomID(next_res.atom_index("N"), i + 1)]
                angle_deg = rosetta.numeric.dihedral_degrees(res.xyz("N"), res.xyz("CA"), res.xyz("C"), next_res.xyz("N"))
                angle_rad = math.radians(angle_deg)
                func = CircularHarmonicFunc(angle_rad, backbone_stddev_rad)
                pose.add_constraint(DihedralConstraint(*ids, func))
            if all([res.has("CA"), res.has("C"), next_res.has("N"), next_res.has("CA")]):
                ids = [AtomID(res.atom_index("CA"), i), AtomID(res.atom_index("C"), i), AtomID(next_res.atom_index("N"), i + 1), AtomID(next_res.atom_index("CA"), i + 1)]
                angle_deg = rosetta.numeric.dihedral_degrees(res.xyz("CA"), res.xyz("C"), next_res.xyz("N"), next_res.xyz("CA"))
                angle_rad = math.radians(angle_deg)
                func = CircularHarmonicFunc(angle_rad, backbone_stddev_rad)
                pose.add_constraint(DihedralConstraint(*ids, func))
    for i in range(1, nres):
        res_i = pose.residue(i)
        res_j = pose.residue(i + 1)
        if res_i.is_protein() and res_j.is_protein():
            if res_i.has("C") and res_j.has("N") and c_n_stddev is not None:
                id1 = AtomID(res_i.atom_index("C"), i)
                id2 = AtomID(res_j.atom_index("N"), i + 1)
                dist = (res_i.xyz("C") - res_j.xyz("N")).norm()
                func = HarmonicFunc(dist, c_n_stddev)
                pose.add_constraint(AtomPairConstraint(id1, id2, func))
    print("Protein constraints added.")

def calculate_per_residue_energies(pose, interface_residues_dict, scorefxn):
    """Calculate per-residue energies for each interface in the pose."""
    nres = pose.total_residue() 
    total_score = scorefxn(pose)
    residue_energies = {}
    for interface_name, value in interface_residues_dict.items():
        residue_indices = value["residues"]
        interface_energies = {}
        for res_idx in residue_indices:
            if res_idx < 1 or res_idx > nres:
                print(f"Warning: residue index {res_idx} out of range for pose with {nres} residues.")
                continue
            residue = pose.residue(res_idx)
            res_name = residue.name3()
            chain_id = pose.pdb_info().chain(res_idx)
            res_num = pose.pdb_info().number(res_idx)
            one_letter = residue.name1()
            try:
                energies = pose.energies()
                res_energy = 0.0
                for score_type in [rosetta.core.scoring.fa_atr, rosetta.core.scoring.fa_rep, 
                                rosetta.core.scoring.fa_sol, rosetta.core.scoring.fa_elec,
                                rosetta.core.scoring.hbond_sr_bb, rosetta.core.scoring.hbond_lr_bb,
                                rosetta.core.scoring.hbond_bb_sc, rosetta.core.scoring.hbond_sc]:
                    try:
                        res_energy += energies.residue_total_energies(res_idx)[score_type]
                    except Exception as e:
                        pass
                interface_energies[res_idx] = {
                    'residue': f"{one_letter}{res_num}",
                    'total_energy': res_energy,
                    'chain': chain_id,
                    'resnum': res_num,
                    'resname': res_name
                }
            except Exception as e:
                print(f"Warning: Could not calculate energy for residue {res_idx}: {e}")
                interface_energies[res_idx] = {
                    'residue': f"{chain_id}{res_num}{res_name}",
                    'total_energy': 0.0,
                    'chain': chain_id,
                    'resnum': res_num,
                    'resname': res_name
                }
        residue_energies[interface_name] = interface_energies
    return residue_energies

def estimate_bsa(pose, chainsA, chainsB, cutoff=5.0, per_contact_bsa=30.0):
    """Estimate the buried surface area (BSA) between two sets of chains by counting close heavy atom contacts. Returns the estimated BSA in Å²."""
    nres = pose.total_residue()
    pdb_info = pose.pdb_info()
    groupA = [i for i in range(1, nres + 1) if pdb_info.chain(i) in chainsA]
    groupB = [i for i in range(1, nres + 1) if pdb_info.chain(i) in chainsB]
    contact_pairs = set()
    for resA in groupA:
        rA = pose.residue(resA)
        for resB in groupB:
            if resA == resB:
                continue
            rB = pose.residue(resB)
            for a1 in range(1, rA.natoms() + 1):
                if rA.atom_name(a1).strip().startswith('H'):
                    continue
                xyz1 = rA.xyz(a1)
                for a2 in range(1, rB.natoms() + 1):
                    if rB.atom_name(a2).strip().startswith('H'):
                        continue
                    xyz2 = rB.xyz(a2)
                    if (xyz1 - xyz2).norm() < cutoff:
                        contact_pairs.add((min(resA, resB), max(resA, resB)))
                        break
                else:
                    continue
                break
    n_contacts = len(contact_pairs)
    estimated_bsa = n_contacts * per_contact_bsa
    return estimated_bsa

def analyze_interface_quality(pose, interface_residues_dict, scorefxn):
    analysis = {}
    for interface_name, value in interface_residues_dict.items():
        residue_indices = value["residues"]
        interface_size = len(residue_indices)
        chainsA, chainsB = value["chains"]
        if set(chainsA) & set(chainsB) or not chainsA or not chainsB:
            estimated_bsa = 0.0
        else:
            estimated_bsa = estimate_bsa(pose, chainsA, chainsB, cutoff=5.0, per_contact_bsa=30.0)
        if interface_name in interface_config:
            ddg, complex_score, separated_score = calculate_interface_ddg(pose, scorefxn, interface_config, interface_name)
        else:
            ddg, complex_score, separated_score = 0.0, 0.0, 0.0
        analysis[interface_name] = {
            "size": interface_size,
            "bsa_true": estimated_bsa,
            "score": ddg,
            "composition": f"{chainsA}-{chainsB}",
        }
    return analysis

def random_vector(magnitude=1.0, normalize=False):
    """Generate a random 3D vector. If normalize=True, returns a unit vector."""
    vec = rosetta.numeric.xyzVector_double_t(
        random.uniform(-magnitude, magnitude),
        random.uniform(-magnitude, magnitude),
        random.uniform(-magnitude, magnitude)
    )
    if normalize:
        vec.normalize()
    return vec

def generate_random_rigid_body_perturbation(translation_mag=None, rotation_mag_deg=None):
    """Generate a random translation vector and rotation matrix for rigid body perturbation. Returns (translation, axis, angle_rad, rot_matrix)"""
    translation_mag = translation_mag if translation_mag is not None else rb_translation_mag
    rotation_mag_deg = rotation_mag_deg if rotation_mag_deg is not None else rb_rotation_mag_deg
    translation = random_vector(translation_mag)
    axis = random_vector(normalize=True)
    angle_rad = math.radians(random.uniform(-rotation_mag_deg, rotation_mag_deg))
    rot_matrix = rosetta.numeric.rotation_matrix(axis, angle_rad)
    return translation, axis, angle_rad, rot_matrix

def calculate_interface_ddg(pose, scorefxn, interface_config, interface_name, separation_distance=None): 
    """Calculate interface ΔΔG for any user-defined interface."""
    if separation_distance is None:
        separation_distance = interface_separation_distance  
    config = interface_config[interface_name]
    group_a = config["group_a"]
    group_b = config["group_b"]
    all_chains = list(set(group_a + group_b))
    pose_chains = set([pose.pdb_info().chain(i) for i in range(1, pose.total_residue() + 1)])

    if set(all_chains) == pose_chains:
        pose_complex = pose.clone()
        pose_complex.remove_constraints()
        complex_score = scorefxn(pose_complex)
        jump_id = None
        for chain in group_b:
            try:
                jump_id = get_jump_to_chain(pose_complex, chain)
                break
            except Exception:
                continue
        if jump_id is None:
            raise ValueError(f"Could not find jump for any chain in group_b: {group_b}")
        pose_sep = pose_complex.clone()
        pose_sep.remove_constraints()
        jump = pose_sep.jump(jump_id)
        jump.set_translation(jump.get_translation() + xyzVector_double_t(separation_distance, 0, 0))
        pose_sep.set_jump(jump_id, jump)
        separated_score = scorefxn(pose_sep)
    else:
        subpose = extract_subpose_by_chains(pose, all_chains)
        pdb_info = subpose.pdb_info()
        chain_a_start = chain_a_end = chain_b_start = chain_b_end = None
        for i in range(1, subpose.total_residue() + 1):
            chain = pdb_info.chain(i)
            if chain in group_a:
                if chain_a_start is None:
                    chain_a_start = i
                chain_a_end = i
            elif chain in group_b:
                if chain_b_start is None:
                    chain_b_start = i
                chain_b_end = i
        chains_present = [pdb_info.chain(i) for i in range(1, subpose.total_residue() + 1)]
        if None in (chain_a_start, chain_a_end, chain_b_start, chain_b_end):
            raise ValueError(
                f"Could not find chains {group_a} and {group_b} in subpose for {interface_name}! "
                f"Found: {group_a}=({chain_a_start},{chain_a_end}), {group_b}=({chain_b_start},{chain_b_end}). "
                f"Chains present: {set(chains_present)}"
            )
        ft = rosetta.core.kinematics.FoldTree()
        ft.add_edge(chain_a_start, chain_a_end, -1)
        ft.add_edge(chain_a_end, chain_b_start, 1)
        ft.add_edge(chain_b_start, chain_b_end, -1)
        subpose.fold_tree(ft)
        pose_complex = subpose.clone()
        pose_complex.remove_constraints()
        complex_score = scorefxn(pose_complex)
        pose_sep = subpose.clone()
        pose_sep.remove_constraints()
        jump = pose_sep.jump(1)
        jump.set_translation(jump.get_translation() + xyzVector_double_t(separation_distance, 0, 0))
        pose_sep.set_jump(1, jump)
        separated_score = scorefxn(pose_sep)
    ddg = complex_score - separated_score
    return ddg, complex_score, separated_score

def get_jump_to_chain(pose, chain_id):
    chain_start = None
    for i in range(1, pose.num_chains() + 1):
        if pose.pdb_info().chain(pose.chain_begin(i)) == chain_id:
            chain_start = pose.chain_begin(i)
            break
    if chain_start is None:
        raise ValueError(f"Chain {chain_id} not found in pose")
    for j in range(1, pose.num_jump() + 1):
        jump = pose.fold_tree().jump_edge(j)
        if jump.stop() == chain_start:
            return j
    raise ValueError(f"No jump found ending at chain {chain_id}")

def detect_residue_types(pose):
    """Detect what types of residues are present in the pose."""
    has_protein = False
    has_nucleic = False
    has_dna = False
    has_rna = False
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        if res.is_protein():
            has_protein = True
        elif res.is_DNA():
            has_nucleic = True
            has_dna = True
        elif res.is_RNA():
            has_nucleic = True
            has_rna = True
    return has_protein, has_nucleic, has_dna, has_rna

def add_conditional_constraints(pose, interface_residues_dict, distance_stddev=None, backbone_distance_cutoff=None):
    """ Add constraints based on detected residue types. """
    if distance_stddev is None:
        distance_stddev = interface_distance_stddev
    if backbone_distance_cutoff is None:
        backbone_distance_cutoff = interface_backbone_distance_cutoff
    has_protein, has_nucleic, has_dna, has_rna = detect_residue_types(pose)
    print(f"Detected residue types: Protein={has_protein}, DNA={has_dna}, RNA={has_rna}")
    if len(interface_residues_dict) > 0:
        add_interface_constraints(pose, interface_residues_dict, distance_stddev, backbone_distance_cutoff)
    if has_protein:
        add_protein_constraints(
            pose,
            angle_stddev=prot_angle_stddev,
            backbone_stddev_deg=prot_backbone_stddev_deg,
            planarity_stddev_deg=prot_planarity_stddev_deg,
            c_n_stddev=prot_c_n_stddev
        )
    else:
        print("No protein residues detected - skipping protein constraints")
    if has_nucleic:
        add_nucleic_acid_constraints(
            pose,
            bond_stddev=nuc_bond_stddev,
            angle_stddev=nuc_angle_stddev,
            pucker_stddev_deg=nuc_pucker_stddev_deg,
            hbond_stddev=nuc_hbond_stddev,
            coplanarity_stddev_deg=nuc_coplanarity_stddev_deg,
            backbone_stddev_deg=nuc_backbone_stddev_deg,
            planarity_stddev_deg=nuc_planarity_stddev_deg,
            critical_angle_stddev=nuc_critical_angle_stddev,
            all_angle_stddev=nuc_all_angle_stddev,
            o3p_p_stddev=nuc_o3p_p_stddev
        )
    else:
        print("No nucleic acid residues detected - skipping nucleic acid constraints")
        
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

n_chains = pose.num_chains()
chain_ids = [pose.pdb_info().chain(pose.chain_begin(i)) for i in range(1, n_chains + 1)]
print(" ")  
print("Detected chain IDs:", chain_ids)

interface_config = interface_config

jumps = {}
for interface_name, config in interface_config.items():
    jumps[interface_name] = get_jump_to_chain(pose, config["jump_chain"])

print(" ") 

for i in range(n_chains):
    chain_id = pose.pdb_info().chain(pose.chain_begin(i+1))
    start = pose.chain_begin(i+1)
    end = pose.chain_end(i+1)
    print(f"Chain {chain_id}: residues {start}-{end}")

all_interface_residues = set()
interface_residues_dict = {}
for i in range(len(chain_ids)):
    for j in range(i + 1, len(chain_ids)):
        residues = get_interface_residues(pose, [
            pose.chain_begin(i + 1),  
            pose.chain_begin(j + 1)
        ])
        key = f"interface_{chain_ids[i]}_{chain_ids[j]}"
        interface_residues_dict[key] = {
            "residues": list(residues),
            "chains": (chain_ids[i], chain_ids[j])
        }
        all_interface_residues.update(residues)

for interface_name, config in interface_config.items():
    group_a = config["group_a"]
    group_b = config["group_b"]
    
    interface_residues = set()
    for c1 in group_a:
        for c2 in group_b:
            key = f"interface_{c1}_{c2}"
            if key in interface_residues_dict:
                for res in interface_residues_dict[key]["residues"]:
                    chain = pose.pdb_info().chain(res)
                    if chain == c1 or chain == c2:
                        interface_residues.add(res)
    
    interface_residues_dict[interface_name] = {
        "residues": list(interface_residues),
        "chains": (group_a, group_b)
    }

interface_residues = sorted(all_interface_residues)
all_interface_res = set()
for interface_name in interface_config.keys():
    all_interface_res.update(interface_residues_dict[interface_name]["residues"])

scorefxn_cart.set_weight(score_type_from_name("fa_atr"), fa_atr_init)
scorefxn_cart.set_weight(score_type_from_name("fa_rep"), fa_rep_init * 1.5) 
scorefxn_cart.set_weight(score_type_from_name("fa_sol"), fa_sol_init)
scorefxn_cart.set_weight(score_type_from_name("fa_intra_rep"), fa_intra_rep_init)
scorefxn_cart.set_weight(score_type_from_name("fa_intra_sol_xover4"), fa_intra_sol_xover4_init)
scorefxn_cart.set_weight(score_type_from_name("lk_ball_wtd"), lk_ball_wtd_init)
scorefxn_cart.set_weight(score_type_from_name("fa_elec"), fa_elec_init)
scorefxn_cart.set_weight(score_type_from_name("hbond_sr_bb"), hbond_sr_bb_init)
scorefxn_cart.set_weight(score_type_from_name("hbond_lr_bb"), hbond_lr_bb_init)
scorefxn_cart.set_weight(score_type_from_name("hbond_bb_sc"), hbond_bb_sc_init)
scorefxn_cart.set_weight(score_type_from_name("hbond_sc"), hbond_sc_init)
scorefxn_cart.set_weight(score_type_from_name("dslf_fa13"), dslf_fa13_init)
scorefxn_cart.set_weight(score_type_from_name("omega"), omega_init)
scorefxn_cart.set_weight(score_type_from_name("fa_dun"), fa_dun_init)
scorefxn_cart.set_weight(score_type_from_name("p_aa_pp"), p_aa_pp_init)
scorefxn_cart.set_weight(score_type_from_name("yhh_planarity"), yhh_planarity_init)
scorefxn_cart.set_weight(score_type_from_name("ref"), ref_init)
scorefxn_cart.set_weight(score_type_from_name("rama_prepro"), rama_prepro_init)
scorefxn_cart.set_weight(rosetta.core.scoring.cart_bonded, cart_bonded_init * 1.5) 
scorefxn_cart.set_weight(rosetta.core.scoring.dihedral_constraint, dihedral_init * 1.25)
scorefxn_cart.set_weight(rosetta.core.scoring.atom_pair_constraint, atom_pair_init)
scorefxn_cart.set_weight(score_type_from_name("rna_torsion"), rna_torsion_init * 1.25)
scorefxn_cart.set_weight(score_type_from_name("rna_sugar_close"), rna_sugar_close_init * 2.0)
scorefxn_cart.set_weight(score_type_from_name("dna_bb_torsion"), dna_bb_torsion_init * 1.25)
scorefxn_cart.set_weight(score_type_from_name("dna_sugar_close"), dna_sugar_close_init * 2.0)
scorefxn_cart.set_weight(rosetta.core.scoring.fa_stack, fa_stack_init)

print(" ")
add_conditional_constraints(pose, interface_residues_dict, distance_stddev=None, backbone_distance_cutoff=None)

initial_pose = pose.clone()

num_clones = num_clones

clone_metrics = []

interface_residues_to_sample = set()
for interface_name, config in interface_config.items():
    interface_residues_to_sample.update(interface_residues_dict[interface_name]["residues"])

interface_selector = ResidueIndexSelector()
interface_selector.set_index(','.join(str(r) for r in sorted(interface_residues_to_sample)))
prevent_noninterface = task_op.PreventRepackingRLT()
restrict_noninterface = task_op.OperateOnResidueSubset(prevent_noninterface, interface_selector, flip_subset=True)

print(" ")

for clone in range(num_clones):

    log_print(f"=== INTERFACE SAMPLING : ITERATION {clone+1}/{num_clones} ===")

    pose_clone = initial_pose.clone()

    for interface_name, config in interface_config.items():
        jump_id = jumps[interface_name]
        translation, axis, angle_rad, rot_matrix = generate_random_rigid_body_perturbation()
        if 1 <= jump_id <= pose_clone.num_jump():
            jump = pose_clone.jump(jump_id)
            jump.set_translation(jump.get_translation() + translation)
            jump.set_rotation(jump.get_rotation() * rot_matrix)
            pose_clone.set_jump(jump_id, jump)

    print(" ")
    print("Interfaces perturbed.")

    log_print(" ")

    movemap = MoveMap()
    movemap.set_bb(False)
    movemap.set_chi(False)
    for res in sorted(interface_residues_to_sample):
        movemap.set_bb(res, True)
        movemap.set_chi(res, True)
    
    interface_chains = set()
    for config in interface_config.values():
        interface_chains.add(config["jump_chain"])
    
    for j in range(1, pose_clone.num_jump() + 1):
        jump = pose_clone.fold_tree().jump_edge(j)
        end_chain = pose_clone.pdb_info().chain(jump.stop())
        if end_chain in interface_chains:
            movemap.set_jump(j, True)

    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    tf.push_back(IncludeCurrent())
    tf.push_back(restrict_noninterface)
    packer_task = tf.create_task_and_apply_taskoperations(pose_clone)
    pack_mover = PackRotamersMover(scorefxn_cart, packer_task)
    pack_mover.apply(pose_clone)

    pose_clone.update_residue_neighbors()

    min_mover_jumps = MinMover()
    min_mover_jumps.movemap(movemap)
    min_mover_jumps.score_function(scorefxn_cart)
    min_mover_jumps.min_type('lbfgs_armijo_nonmonotone')
    min_mover_jumps.max_iter(min_max_iter)
    min_mover_jumps.tolerance(min_tolerance)
    min_mover_jumps.cartesian(True)
    min_mover_jumps.apply(pose_clone)

    pose_clone.update_residue_neighbors()

    pack_mover = PackRotamersMover(scorefxn_cart, packer_task)
    pack_mover.apply(pose_clone)

    pose_clone.remove_constraints()
    pose_clone.energies().clear()
    score_cart = scorefxn_cart(pose_clone)
    print(" ")
    print(f"Score = {score_cart:.3f}")

    interface_analysis = analyze_interface_quality(pose_clone, interface_residues_dict, scorefxn_cart)
    residue_energies = calculate_per_residue_energies(pose_clone, interface_residues_dict, scorefxn_cart)

    clone_metrics_entry = {
        "clone": clone + 1,
        "score": score_cart,
    }

    interface_subpose_scores = {}
    for interface_name, config in interface_config.items():
        ddg, complex_score, separated_score = calculate_interface_ddg(
            pose_clone, scorefxn_cart, interface_config, interface_name
        )
        
        interface_subpose_scores[interface_name] = {
            'interface_ddg': ddg,
            'complex_score': complex_score,
            'separated_score': separated_score,
            'binding_energy': ddg  
        }

        clone_metrics_entry[f"{interface_name}_interface_ddg"] = ddg
        clone_metrics_entry[f"{interface_name}_complex_score"] = complex_score
        clone_metrics_entry[f"{interface_name}_separated_score"] = separated_score

        bsa = interface_analysis[interface_name]["bsa_true"] if interface_name in interface_analysis else None
        clone_metrics_entry[f"{interface_name}_bsa"] = bsa

    clone_metrics.append(clone_metrics_entry)

    print(" ")

    for interface_name, config in interface_config.items():
        group_a = config["group_a"]
        group_b = config["group_b"]
        group_a_pose = extract_subpose_by_chains(pose_clone, group_a)
        group_b_pose = extract_subpose_by_chains(pose_clone, group_b)
        group_a_score = scorefxn_cart(group_a_pose)
        group_b_score = scorefxn_cart(group_b_pose)
        interface_subpose_scores[interface_name]['group_a_score'] = group_a_score
        interface_subpose_scores[interface_name]['group_b_score'] = group_b_score

    log_print(f"{chain_ids}: {score_cart:.2f}")

    for interface_name, config in interface_config.items():
        group_a = config["group_a"]
        group_b = config["group_b"]
        group_a_score = interface_subpose_scores[interface_name]['group_a_score']
        group_b_score = interface_subpose_scores[interface_name]['group_b_score']
        log_print(f"{group_a} submodel score: {group_a_score:.2f}")
        log_print(f"{group_b} submodel score: {group_b_score:.2f}")

    for interface_name in interface_config.keys():
        scores = interface_subpose_scores[interface_name]
        interface_ddg = scores['interface_ddg']
        complex_score = scores['complex_score']
        separated_score = scores['separated_score']
        log_print(f"{interface_name} ΔΔG: {interface_ddg:.2f} (complex: {complex_score:.2f}, separated: {separated_score:.2f})")
        
    for interface_name in interface_config.keys():
        log_print(" ")
        section_title = f"{interface_name.upper()} - PER-RESIDUE ENERGY CONTRIBUTION:"
        log_print(section_title)
        log_print("")
        log_print("Rank Chain Residue Energy")
        log_print("")
        energies = residue_energies[interface_name]
        sorted_residues = sorted(energies.items(), key=lambda x: x[1]['total_energy'])
        for i, (res_idx, data) in enumerate(sorted_residues):
            log_print(f"{str(i+1):<4} {data['chain']:<5} {data['residue']:<7} {str(round(data['total_energy'],2)):<7}")

        log_print("")

        if interface_analysis and interface_name in interface_analysis:
            data = interface_analysis[interface_name]
            comp = residue_composition(interface_residues_dict[interface_name]["residues"], pose_clone)
            comp_str = f"hydrophobic: {comp['hydrophobic']}, polar: {comp['polar']}, charged: {comp['charged']}"
            log_print(f"{interface_name}:")
            log_print(f"Size: {data['size']} residues")
            log_print(f"Estimated BSA: {data['bsa_true']:.1f} Å²")
            log_print(f"Composition: {comp_str}")

    log_print(" ")