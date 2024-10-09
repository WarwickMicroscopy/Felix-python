def read_hkl_file(filename):
    """
    Reads in the file felix.hkl and returns the list of reflexions to output

    Parameters:
    filename (str): The path to the input file.
    Returns:
    inp_dict: A dictionary with variable names and values.
    """

    input_hkls = []
    i_obs = []
    sigma_obs = []

    cRED = False  # cRED data has Iobs & sigma as extra columns
    with open(filename, 'r') as file:
        # Check the first line to decide the file structure
        first_line = file.readline().strip()
        if ',' in first_line.split(']')[-1]:  # comma after ]
            cRED = True
        # Go back to the beginning of the file
        file.seek(0)

        for line in file:
            # Remove brackets and split the line
            line = line.strip().replace('[', '').replace(']', '')
            parts = line.split()

            # Extract Miller indices
            vector_part = parts[0]
            hkl = list(map(int, vector_part.split(',')))

            if cRED:  # Extract i_obs and sigma_obs
                intensity = float(parts[1].replace(',', ''))
                sigma = float(parts[2])
                i_obs.append(intensity)
                sigma_obs.append(sigma)

            # Append to respective lists
            input_hkls.append(hkl)
            i_obs.append(intensity)
            sigma_obs.append(sigma)

    # Convert lists to numpy arrays
    input_hkls = np.array(input_hkls)
    i_obs = np.array(i_obs) if cRED else None
    sigma_obs = np.array(sigma_obs) if cRED else None

    return input_hkls, i_obs, sigma_obs
