import os
import gemmi
import argparse
import gzip
import json
import pickle
import itertools
import numpy as np
from pathlib import Path
from math import exp
import openstructure as op

DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), 'data')
LIBRARY_PATH = os.path.join(DATA_DIR_PATH, 'library.gz')
library_data = None

def product(x):
    result = 1
    for x_i in x:
        result *= x_i
    return result

def unpack_bytes(in_bytes):
    try:
        masks = np.array([ 0b11000000, 0b00110000, 0b00001100, 0b00000011 ])
        shifts = np.array([ 6, 4, 2, 0 ])
        masked = np.array(in_bytes).reshape(-1, 1) & np.array(masks)
        shifted = masked >> np.array(shifts)
        unpacked = shifted.flatten().astype('int8')
    except ImportError:
        # Python-only mode
        bits = list(itertools.product(range(4), repeat=4))
        replaced = [ bits[b] for b in in_bytes ]
        unpacked = [ bits for byte in replaced for bits in byte ]
    return unpacked

def load_rotamer_data():
        with gzip.open(LIBRARY_PATH, 'rb') as infile:
            dim_offsets, dim_bin_ranges, dim_bin_widths, dim_num_options, compressed_byte_arrays = pickle.load(infile)
        classifications = { }
        for code, compressed in compressed_byte_arrays.items():
            compressed = bytearray(compressed)
            classifications[code] = unpack_bytes(compressed)
        library_data = (dim_offsets, dim_bin_ranges, dim_bin_widths, dim_num_options, classifications)

def get_classification(code, chis):
    dim_offsets, dim_bin_ranges, dim_bin_widths, dim_num_options, classifications = library_data
    if  None in chis:
        return
    if code not in dim_offsets.keys():
        return
    closest_values = [ ]
    chis = tuple([ x for x in chis if x is not None ][:len(dim_offsets[code])])
    for dimension, chi in enumerate(chis):
        dim_width = dim_bin_ranges[code][dimension][1] - dim_bin_ranges[code][dimension][0]
        if chi <= dim_bin_ranges[code][dimension][0]:
            chi += dim_width
        if chi >= dim_bin_ranges[code][dimension][1]:
            chi -= dim_width
        multiple = round((chi - dim_offsets[code][dimension]) / dim_bin_widths[code][dimension])
        closest_value = dim_offsets[code][dimension] + multiple * dim_bin_widths[code][dimension]
        closest_values.append(closest_value)
    closest_values = tuple(closest_values)
    index = 0
    for dimension, chi in enumerate(closest_values):
        dim_offest = dim_offsets[code][dimension]
        dim_bin_width = dim_bin_widths[code][dimension]
        index += int((chi - dim_offest) / dim_bin_width * product(dim_num_options[code][dimension+1:]))
    return classifications[code][index]


def plddt_to_rmsd ( plddt = 0.0 ) :
  frac_lddt = plddt / 100.0
  rmsd_estimation = 1.5 * exp(4.0*(0.7-frac_lddt))
  return rmsd_estimation


def plddt_to_bfact ( plddt = 0.0 ) :
  return min ( 999.99, 26.318945069571623 * (plddt_to_rmsd ( plddt ))**2)




def create_script_file ( filename = "", list_of_hits = [ ] ) :
  with open ( filename.split('.')[0] + '.py', 'w' ) as file_out :
    file_out.write ( "# File programmatically created by Fletcher\n" )
    file_out.write ( 'handle_read_draw_molecule_with_recentre ("%s", 1)\n' % filename )
    file_out.write ( 'interesting_things_gui ("Results from Fletcher",[\n')
    for hit in list_of_hits :
      file_out.write ( '["%s %s", %.3f, %.3f, %.3f, ]' \
                                % ( hit[0].get('name'), \
                                    hit[0].get('seqid'), \
                                    hit[0].get('coordinates')[0], \
                                    hit[0].get('coordinates')[1], \
                                    hit[0].get('coordinates')[2] ))
      if hit is not list_of_hits[-1] :
        file_out.write(',\n')
    file_out.write ( '])\n')
    file_out.close ( )

def calculate_lddt_openstructure(af_model_path, ref_model_path, residue):
    """Calculates LDDT using OpenStructure."""
    try:
        af_structure = op.read_pdb(af_model_path)
        ref_structure = op.read_pdb(ref_model_path)

        # Find corresponding residues (important to handle chain and residue number/name correctly)
        af_chain_id = residue.get_parent().id
        af_res_seqid = residue.get_id()[1]
        af_res_name = residue.get_resname()

        ref_residue = None
        for ref_chain in ref_structure.chains:
            if ref_chain.id == af_chain_id:
                for ref_res in ref_chain.residues:
                    if ref_res.name == af_res_name and ref_res.id == af_res_seqid:
                        ref_residue = ref_res
                        break
                if ref_residue is not None:
                  break
        if ref_residue is None:
            return None # Couldn't find matching residue

        # Calculate LDDT
        lddt_calculator = op.LDDTCalculator()
        lddt_score = lddt_calculator.calculate(af_structure, ref_structure, [residue.get_id()[1]], chain_identifier = af_chain_id)[0] #Pass the residue number, not the whole residue

        return lddt_score * 100  # Convert to percentage

    except Exception as e:  # Catch potential errors (file issues, etc.)
        print(f"Error in LDDT calculation: {e}")
        return None


def find_matching_residue(ref_model, residue):
    """Finds the corresponding residue in the reference model based on chain ID and sequence ID."""
    for ref_chain in ref_model:  # Iterate through chains in the reference model
        if ref_chain.id == residue.get_parent().id: # Check if the chains are the same
          for ref_residue in ref_chain:
            if ref_residue.get_resname() == residue.get_resname() and ref_residue.get_id()[1] == residue.get_id()[1]:  # Compare seqid and residue name
                return ref_residue
    return None  # Return None if no match is found


def find_structural_motifs ( filename = "",
                             residue_lists = [ ],
                             distance = 0.0,
                             min_plddt = 70.0,
                             n_term = False,
                             c_term = False,
                             reference = "",
                            ) :
                                
  ref_model = gemmi.read_structure(reference)  # Load the reference structure
  af_model = gemmi.read_structure ( filename )
  neighbour_search = gemmi.NeighborSearch ( af_model[0], af_model.cell, distance ).populate ( include_h=False )
  first_residues = gemmi.Selection ( '(' + residue_lists[0][0] + ')' ) 
  
  result_dict = { }
  result_list = [ ]

  for model in first_residues.models(af_model):
    for chain in first_residues.chains(model):
      for residue in first_residues.residues(chain):
        partial_result = [ residue ]
        marks = neighbour_search.find_neighbors ( residue[-1], 0, distance )
        for candidate_list in residue_lists[1:] :
          for candidate in candidate_list :
            found_in_contacts = False
            for mark in marks :
              cra = mark.to_cra ( af_model[0] )
              
              # We do the following conversion to harness gemmi's translation of modified residue codes
              # into the unmodified ones, e.g. HIC (methylated histidine) >> HIS (normal histidine)
              if gemmi.find_tabulated_residue(candidate).one_letter_code.upper() == \
                 gemmi.find_tabulated_residue(cra.residue.name).one_letter_code.upper() \
                 and cra.residue not in partial_result :
                
                partial_result.append ( cra.residue )
                found_in_contacts = True
                break
            if found_in_contacts :
              break
          if len(residue_lists) == len(partial_result) :
            if (n_term or c_term) :
              in_terminus = False
              for residue in partial_result :
                if n_term and residue == chain[0] :
                  in_terminus = True
                elif c_term and residue.seqid.num == chain[-1].seqid.num :
                  in_terminus = True
              if in_terminus : result_list.append ( partial_result )
            else :
              result_list.append ( partial_result )
            
  if len ( result_list ) > 0 :
    Path ( filename ).touch() # We want results at the top
    result_dict['filename'] = filename
    result_dict['residue_lists'] = str(residue_lists)
    result_dict['distance'] = distance
    result_dict['plddt'] = min_plddt
    hit_list = [ ]

    for result in result_list :
      hit = [ ]
      for residue in result :
        residue_dict = { }
        residue_dict['name']  = residue.name
        residue_dict['seqid'] = str(residue.seqid)
        if residue[-1].b_iso < min_plddt :
          residue_dict['plddt'] = 'LOW PLDDT: %.2f' % residue[-1].b_iso
        else :
          residue_dict['plddt'] = '%.2f' % residue[-1].b_iso
        residue_dict ['coordinates'] = residue[-1].pos.tolist()

      # LDDT calculation
      ref_residue = find_matching_residue(ref_model[0], residue)
     ref_residue = find_matching_residue(ref_model[0], residue)  # Correct indentation here
            if ref_residue:  # Correct indentation here
                lddt_score = calculate_lddt_openstructure(filename, reference, residue)
                if lddt_score is not None:
                    residue_dict['lddt'] = "%.2f" % lddt_score
                else:
                    residue_dict['lddt'] = "N/A"
            else:
                residue_dict['lddt'] = "N/A"

            hit.append(residue_dict)
        hit_list.append(hit)
        print("Hit found:", hit)

    result_dict['hits'] = hit_list

    with open ( filename.split('.')[0] + '.json', 'w' ) as file_out :
      json.dump ( result_dict, file_out, sort_keys=False, indent=4 )
    
    create_script_file ( filename, hit_list )
  
  else :
    print ("\nNo results found :-( \n")
  return result_dict

if __name__ == '__main__':
  parser = argparse.ArgumentParser ( 
                    prog='Fletcher',
                    description='Fletcher will try to find a list of residues within a fixed distance from the centre of mass.'\
                                '\nConcept: Federico Sabbadin & Jon Agirre, University of York, UK.',
                    epilog='Please send bug reports to Jon Agirre: jon.agirre@york.ac.uk' )

  parser.add_argument ( '-f', '--filename', \
                        help = "The name of the file to be processed, in PDB or mmCIF format.", \
                        required = True )                  

  parser.add_argument ( '-r', '--residues', \
                        help = "A list of residues in one-letter code, comma separated, and including alternatives, e.g. L,A,FWY.", \
                        default = "GF", required = True )                       

  parser.add_argument ( '-d', '--distance', \
                        help = "Specifies how far each of the residues can be from the rest, in Angstroems.", \
                        default = "0.0", required = True )  

  parser.add_argument ( '-p', '--plddt', \
                        help = "Flag up candidate residues with average pLDDT below thresold (Jumper et al., 2020).", \
                        default = "70.0", required = False )
  
  parser.add_argument ( '-n', '--nterm', \
                        help = 'Require one residue to be at the n-terminus', \
                        choices = [ 'yes', 'no' ], \
                        default = 'no' )
  
  parser.add_argument ( '-c', '--cterm', \
                        help = 'Require one residue to be at the c-terminus', \
                        choices = [ 'yes', 'no' ], \
                        default = 'no' )
    
  parser.add_argument ( '-ref', '--reference', \
                       help = "Reference Model for LDDT comparison.", \
                       required = True )
    
  args = parser.parse_args ( )
  
  # Assuming argparse has got the right number of parameters beyond this point

  print ( "\nFletcher is a tool that helps spot and document molecular features in AlphaFold models."\
          "\nConcept: Federico Sabbaddin & Jon Agirre, University of York, UK."\
          "\nLatest source code: https://github.com/glycojones/fletcher"\
          "\nBug reports to jon.agirre@york.ac.uk\n\n" )

  input_residues = args.residues.split(',')
  list_of_residues = [ ]

  for slot in input_residues :
    list_of_residues.append ( gemmi.expand_one_letter_sequence(slot, gemmi.ResidueKind.AA) )

  distance = float ( args.distance )
  min_plddt = float ( args.plddt )
  n_term = True if args.nterm == 'yes' else False
  c_term = True if args.cterm == 'yes' else False

  print ( "Running Fletcher with the following parameters:\nFilename: ", 
          args.filename, "\nResidue list: ", 
          list_of_residues, "\nDistance: ", 
          distance, "\npLDDT: ",
          min_plddt,
          "\nN-term: ", n_term,
          "\nC-term: ", c_term,
          "\nReference Model: ", args.reference,
          "\n" )
  
  if len ( list_of_residues ) > 1 and distance > 0.0 :
    find_structural_motifs ( args.filename, list_of_residues, distance, min_plddt, n_term, c_term, args.reference )

