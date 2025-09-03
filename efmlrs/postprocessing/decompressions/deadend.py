def parse_info(file):
    """
    Reads decompression information from infofile and builds decompression list for deadend compressions.

    :param file: infofile automatically created during compressions, containing all information for decompressions
    :return: cmp_info: list containing decompression information for deadend compressions
    """
    cmp_infos = []
    for line in file:
        if line.startswith("remove"):
            reactions = []
            for rea in line.split()[1:]:
                reactions.append(int(rea))
            cmp_infos.append(reactions)
        if line.startswith("end"):
            break
    return cmp_infos


def decompressions(compressed, mapping):
    """
    Decompresses the part of the current compressed efm that has been compressed during deadend compression by inserting
    zeros at the index of deadend metabolites.

    :param list compressed: list containing efm for decompression
    :param mapping: list containing decompression information for deadend decompressions
    :return: compressed: list containing current efm with decompressed deadend part
    """
    [compressed.insert(val, 0) for val in mapping]
    return compressed
