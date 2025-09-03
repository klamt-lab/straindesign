from fractions import Fraction


def parse_info(file):
    """
    Reads decompression information from infofile and builds decompression list for nullspace compressions.

    :param file: infofile automatically created during compressions, containing all information for decompressions
    :return: cmp_info: list containing decompression information for deadend compressions
    """
    cmp_infos = []
    for line in file:
        if line.startswith("R"):
            infos = {}
            data = line.strip().split()
            for i in data:
                tokens = i.split(",")
                if len(tokens) <= 1:
                    continue
                index_1 = tokens[0][1:]
                index_2, factor = tokens[1].split(":")
                infos[int(index_1)] = (int(index_2), Fraction(factor))
            cmp_infos.append((infos, len(data), len(data) + len(infos)))
        if line.startswith("end"):
            break
    return cmp_infos


def build_mapping(infos, rea_uncomp):
    """
    Builds list containing decompression information for nullspace decompressions.

    :param dictionary infos: dictionary containing index of merged reaction, index of original reaction and merge factor
    :param int rea_uncomp: int number equal to the amount of original reactions
    :return: mapping: list containing decompression information for nullspace decompressions
    """
    mapping = {}

    for index_1, value in infos.items():
        index_2, factor = value
        mapping[index_2] = (index_1, Fraction(factor))

    count = 0
    for i in range(0, rea_uncomp):
        if i in mapping:
            continue
        mapping[i] = count
        count += 1

    mapping = sorted(mapping.items())
    return mapping


def decompressions(cmp_efm, mapping):
    """
    Decompresses the part of the current compressed efm that has been compressed during nullspace compression. This is
    done by splitting reactions that have been merged during nullspace into two reactions. Therefore the merged reaction
    is divided through the merge factor that has been used during nullspace compressions.

    :param list cmp_efm: list containing efm for decompression
    :param mapping: list of tuples containing decompression information for nullspace decompressions
    :return: decmp_efm list containing current efm with decompressed nullspace part
    """
    decmp_efm = []

    for i, val in mapping:
        if type(val) == int:
            decmp_efm.append(cmp_efm[val])
        else:
            index = val[0]
            factor = val[1]
            mapped_index = mapping[index][1]
            new_val = Fraction(cmp_efm[mapped_index]) / factor
            decmp_efm.append(new_val)
    return decmp_efm
