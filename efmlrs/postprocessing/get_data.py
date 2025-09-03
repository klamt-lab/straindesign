def get_rev_info(infofile):
    """
    Reads info file and stores reaction reversibility information in a list.

    :param infofile: info file created during preprocessing
    :return: list rev_info
    """
    file = open(infofile, "r")
    rev_info = []
    for line in file:
        if line.startswith("rv:"):
            rv = line.strip().split(":")[1]
            for val in rv.replace(" ", ""):
                if int(val) == 1:
                    rev_info.append(True)
                else:
                    rev_info.append(False)
    file.close()
    return rev_info


def remove_zeros(row):
    """
    Checks if extracted efm (row) contains only zeros.

    :param list row: extracted efm
    :return: row or None
    """
    for val in row:
        if val != 0:
            return row
    return None


def parse_lrs(inputfile, reversibilities):
    """
    Parses mplrs output file, merges splitted reactions, removes rows containing only zeros and stores compressed efms
    in a list of lists.

    :param inputfile: mplrs output file
    :param list reversibilities: list of reaction reversibilities
    :return: list compressed_efms - list of lists with compressed efms
    """
    mplrs_output = open(inputfile, "r")
    compressed_efms = []
    zero_counter = 0
    found_begin = False

    for line in mplrs_output:
        if line == "":
            continue
        if line.startswith("*"):
            continue
        if line.startswith("begin"):
            found_begin = True
            continue
        if found_begin == False:
            continue
        break

    for line in mplrs_output:
        if line.startswith("end"):
            break
        val = line.split()
        row = []

        j = 0
        i = 1
        while i < len(val):
            if reversibilities[j] == 0:
                row.append(int(val[i]))
            else:
                tmp = int(val[i]) - int(val[i + 1])
                row.append(tmp)
                i += 1
            j += 1
            i += 1

        efm = remove_zeros(row)

        if efm is None:
            zero_counter += 1
        elif efm is not None:
            compressed_efms.append(efm)
            if len(compressed_efms) % 10000 == 0:
                print("EFMS extracted:" + str(len(compressed_efms)))

    mplrs_output.close()
    return compressed_efms


def get_mplrs_efms(inputfile, infofile):
    """
    Entry poitn for get_data for mplrs. Parses mplrs output file, merges splitted reactions, removes rows containing
    only zeros and stores compressed efms in a list of lists.

    :param inputfile: mplrs output file
    :param infofile: info file created during preprocessing
    :return: list compressed_efms
    """
    reversibilities = get_rev_info(infofile)
    compressed_efms = parse_lrs(inputfile, reversibilities)
    return compressed_efms


def get_efmtool_efms(inputfile):
    """
    Entry point for get_data for efmtool. Reads efmtool output file and stores compressed efms in a list of lists.

    :param inputfile: efmtool output file
    :return: list compressed_efms
    """
    ifile = open(inputfile, "r")
    compressed_efms = []
    for line in ifile:
        cmp_efm = [float(val) for val in line.strip().split()]
        compressed_efms.append(cmp_efm)
    return compressed_efms
