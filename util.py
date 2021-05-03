from rdkit.Chem import AllChem
from compound_tools import mol_to_graph_data_obj_simple
from torch_geometric.data import Data

def gen_features(smiles, property):
        
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data = mol_to_graph_data_obj_simple(mol, property)
        return data