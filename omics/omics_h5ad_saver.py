# Importing the neccecary modules
import argparse
import anndata
#import scanpy
import os
import numpy
import scipy.sparse as sp

# Script to split an .h5ad file into a set of 1D numpy arrays,
# append class (cell_type) label and save as individual files.


def matrix_split_save(sc_matrix, cell_types, input_arrays_path):
    """
        Split obs (cells) from a sparce matrix (extracted from a .h5ad file)
        and save into individual .npy files.

        Parameters:
        -----------
        sc_matrix : scipy.sparse.csr_matrix
            Sparce matrix data (from a .h5ad file) to be split and saved.
        cell_types : list
            A List containing class labels (cell type) for each sample (cell).
        input_arrays_path : str
            Path to the directory where the data will be saved.

        Returns:
        --------
        None (files are saved to input_arrays_path directory)

        Notes:
        ------
        This function splits the input matrix `sc_matrix` into 
        individual numpy arrays based on sample classes provided in `cell_types`
        and saves them in separate files in the specified directory.
        Each saved file is named using the sample's class label and sample ID.

        This function was designed for use with files downloaded from cellXgene.

        Example:
        --------
        sc_matrix = ...  # Load your omic matrix data
        cell_types = ['Class1', 'Class2', 'Class1', ...]  # Class labels for each sample
        input_arrays_path = '/path/to/output'  # Directory where the data will be saved
        matrix_split_save(sc_matrix, cell_types, input_arrays_path)
        """

    # Make sure that the user has a directory named 'input_arrays_path'.
    # if it exists, stop with error
    if os.path.exists(input_arrays_path):
        print(f"Error: '{input_arrays_path}' directory already exists. Please specify a different path or remove the directory.")
        exit(1)
    else:
        os.makedirs(input_arrays_path)
        print(f"Directory '{input_arrays_path}' created.")

    # Get cell IDs from the index of the cell_types Series
    cell_names = cell_types.index
    
    # Check if the matrix is sparse. We'll use this to decide how to convert each row.
    is_sparse = sp.issparse(sc_matrix)
    
    total_cells = sc_matrix.shape[0]
    print(f"Starting to save {total_cells} individual cell arrays...")

    # Iterate row by row. This is the memory-efficient part.
    # We zip the cell IDs (sid) and labels together.
    for idx, (sid, label) in enumerate(zip(cell_names, cell_types)):
        
        # 1. Get the single row from the matrix.
        # Slicing a sparse matrix [idx, :] returns a (1, n_features) sparse matrix.
        row_data = sc_matrix[idx, :]

        # 2. Convert *only this row* to a dense 1D array.
        # This is very cheap and uses almost no memory.
        if is_sparse:
            sample = row_data.toarray().flatten() # .toarray() on one row is fine
        else:
            sample = numpy.array(row_data).flatten() # Handle if matrix was already dense

        # 3. Create the file name
        sample_name = str(label) + '_' + str(sid) + '.npy'

        # 4. Build save path
        save_path = os.path.join(input_arrays_path, sample_name)
        
        # 5. Save the 1D dense array
        numpy.save(save_path, sample)

        # 6. Report progress
        # Use (idx + 1) for 1-based counting
        if (idx + 1) % (total_cells // 4) == 0:
            print(f"Saved sample {idx + 1}/{total_cells}")

    # Check that the number of files in input_arrays_path is equal to the number of saved samples
    num_files_saved = len(os.listdir(input_arrays_path))
    if num_files_saved == total_cells:
        print(f"All {num_files_saved} samples saved successfully.")
    else:
        print(f"Warning: Expected {total_cells} samples, but only {num_files_saved} files were saved.")




# parse arguments to extract file paths for saving down the data
if __name__ == '__main__':

    # Creating an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Adding arguments for the script
    # Argument for the path to the .h5ad file
    parser.add_argument(
        "--h5ad_file",
        help="Path to h5ad file, \n" \
            "(includes counts matrix and cell_type column in the metadata)"
    )

    # Argument for the output path to save data (input_arrays)
    parser.add_argument(
        "--input_arrays_path",
        required=True,
        help="Path to save individual sample arrays (.npy files)"
    )

    # Argument for the class list CSV file
    parser.add_argument(
        "--class_lst_path",
        required=True,
        help="Path to save the unique class list CSV file"
    )

    # Argument for the col name in the metadata that contains the cell-type info
    parser.add_argument(
        "--cell_type_column_name",
        default="cell_type",
        help="The col name in metadata that contains the cell-type info, \n" \
        "Default is 'cell_type'"
    )

    # # Argument for the number of most variable genes to include in the analysis
    # parser.add_argument(
    #     "--ngenes",
    #     default=0,
    #     help="The integer of the most vaiable genes to be included, \n" \
    #          "Default is '0' (all genes)"
    # )

    # Parsing the command-line arguments
    args = parser.parse_args()
    print("Parsing command-line arguments...")

    # Assigning values from parsed arguments to variables
    h5ad_file = args.h5ad_file
    input_arrays_path = args.input_arrays_path
    class_lst_path = args.class_lst_path
    cell_type_column_name = args.cell_type_column_name
    # ngenes = int(args.ngenes)

    print(f"Reading .h5ad file from: {h5ad_file}")

    # read in the .h5ad file
    adata = anndata.read_h5ad(h5ad_file)
    print(r"Read in the .h5ad file")

    # If --ngenes is > 0, subset the data to the top ngenes
    # if ngenes > 0:
    #     print(f"Subsetting the data to the top {ngenes} most variable genes")
    #     # identify the most variable genes
    #     scanpy.pp.highly_variable_genes(adata, n_top_genes=ngenes, flavor='cell_ranger')
    #     # this could be skipped by using subset = True above
    #     # downsample to only include the most variable genes
    #     adata = adata[:, adata.var.highly_variable]
    #     print(f"Subsetted the data to the top {ngenes} highly variable genes")
    # else:
    #     print("Using all genes (no subsetting).")
    
    # Extract the counts matrix
    sc_matrix = adata.X
    print(r"Extracted the counts matrix")
    # convert to numpy array if it's a sparse matrix
    # dont convert to array if the matrix is too large
    # RAM usage will be too high
    # if sp.issparse(sc_matrix):
    #     sc_matrix = sc_matrix.toarray()


    # Extract the cell_type column from the metadata
    print(f"Extracting cell type column '{cell_type_column_name}' from metadata...")
    cell_types = adata.obs[cell_type_column_name]
    print(f"Extracted cell type column from metadata {cell_types}")
    # Modify cell_types to remove commas
    cell_types = cell_types.str.replace(', ', '--') # avoided _ as they're buggy
    # Modify cell_types to remove spaces
    cell_types = cell_types.str.replace(' ', '-')
    print(f"Extracted clean cell type column from metadata {cell_types}") 
    # Identify the unique classes
    print("Identifying unique classes...")
    cell_types_unique = numpy.unique(cell_types).reshape(1, -1)
    print(r"Unique classes:" + str(cell_types_unique))

    # Check if the class_lst.csv file already exists
    print("Checking for existing class_lst.csv file...")
    if os.path.exists(class_lst_path):
        print(f"Error: '{class_lst_path}' already exists. Please specify a different path or remove the file.")
        exit(1)
    else:
        # save the unique list of classes as a csv file
        print("Saving unique class list to CSV...")
        numpy.savetxt(class_lst_path, 
                    cell_types_unique, 
                    fmt='%s',  
                    delimiter=",")
        print(f"Unique classes saved to {class_lst_path}")

    # split matrix and save as individual files
    print("Splitting matrix and saving as individual files...")
    matrix_split_save(sc_matrix, cell_types, input_arrays_path)
    print("Done.")
