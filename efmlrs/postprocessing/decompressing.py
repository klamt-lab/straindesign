import efmlrs.postprocessing.decompressions.many2one as many2one
import efmlrs.postprocessing.decompressions.nullspace as nullspace
import efmlrs.postprocessing.decompressions.deadend as deadend


def find_counter(infofile):
    """
    Parses compression info and stores information on how many compression rounds were done during preprocessing
    and how many additional bounds have been applied.

    :param infofile: file automatically created during compressions, containing all information for decompressions
    :return:
        - round_counter - int number of compression rounds
        - bounds - int number of bounds
    """
    round_counter = 0
    bounds = 0
    file = open(infofile, "r")
    for line in file:
        if line.startswith("bounds"):
            bounds = line[7]
        if line.startswith("counter"):
            round_counter = line[8]
    file.close()
    return int(round_counter), int(bounds)


def build_reverse_mapping(infofile, counter):
    """
    Reads compression information form infofile and builds a mapping for deompressions which is in reversed order of
    the previously applied compressions.

    :param infofile: file automatically created during compressions, containing all information for decompressions
    :param int counter: number of compression rounds
    :return: mappings - list of different tuple with compression information for each compression step
    """
    mappings = []
    for i in reversed(range(1, counter + 1)):
        file = open(infofile, "r")
        tmp = []
        DE = False
        O2M = False
        NS = False

        for line in file:
            if DE is True and O2M is True and NS is True:
                break
            if line.startswith("deadend_" + str(i)):
                DE = True
                deadend_cmps = deadend.parse_info(file)
                if len(deadend_cmps) != 0:
                    for reactions in deadend_cmps:
                        tmp.append(("deadend", reactions))

            if line.startswith("many2one_" + str(i)):
                O2M = True
                iterations, post, pre = many2one.parse_info(file)
                if post != pre:
                    rea_mapping = many2one.build_merge_mapping(iterations, post)
                    tmp.append(("m2o", (rea_mapping, iterations, post)))

            if line.startswith("nullspace_" + str(i)):
                NS = True
                null_cmps = nullspace.parse_info(file)
                if len(null_cmps) != 0:
                    for infos, rea_comp, rea_uncomp in reversed(null_cmps):
                        rea_mapping = nullspace.build_mapping(infos, rea_uncomp)
                        tmp.append(("nullspace", rea_mapping))

        for element in reversed(tmp):
            mappings.append(element)

        file.close()
    return mappings


def normalize_efms(decompressed, bound_info):
    """
    Only called if model had additional boundaries. Removes lambda vector entry and additional boundary reactions from
    current efm and normalizes it.

    :param list decompressed: current efm as list
    :param int bound_info: number of additional bounds
    :return: decompressed: current efm as list
    """
    lambda_val = decompressed[-1]
    del decompressed[-(bound_info + 1):]
    if lambda_val != 1 and lambda_val != 0:
        new_compressed = [val / lambda_val for val in decompressed]
        return new_compressed
    else:
        return decompressed


def write_decompressed_efms(decompressed, outputfile):
    """
    Writes final decompressed efm to user specified output file.

    :param decompressed: current efm as list
    :param outputfile: user specified filed to write decompressed files in
    :return: None
    """
    for val in decompressed:
        val = float(val)
        outputfile.write(str(val) + " ")
    outputfile.write("\n")


def decompressing(compressed_efms, outputfile, mappings, bound_info):
    """
    Iteratively decompresses one efm after another by applying decompressions according to decompression information
    stored in mappings. If model had additional boundaries: removes lambda vector entry and additional boundary
    reactions from current efm and normalizes it. Writes final decompressed efm to user specified output file.

    :param list compressed_efms: list of lists containing compressed efms
    :param outputfile: user specified filed to write decompressed files in
    :param mappings: list of different tuples with compression information for each compression step
    :param int bound_info: number of additional bounds
    :return: None
    """
    ofile = open(outputfile, "w")
    count = 0

    for cmp_efm in compressed_efms:
        decompressed = cmp_efm
        for infotype, mappinginfo in mappings:
            if infotype == "nullspace":
                decompressed = nullspace.decompressions(decompressed, mappinginfo)

            elif infotype == "m2o":
                mapping, iterations, post = mappinginfo
                decompressed = many2one.decompressions(decompressed, mapping, iterations, post)

            elif infotype == "deadend":
                decompressed = deadend.decompressions(decompressed, mappinginfo)
                continue

        if bound_info != 0:
            normalized_efms = normalize_efms(decompressed, bound_info)
            write_decompressed_efms(normalized_efms, ofile)
        else:
            write_decompressed_efms(decompressed, ofile)

        count += 1
        if count % 1000 == 0:
            print("EFMs decompressed:", count)
    ofile.close()
    print("Decompressed EFMs:", count)


def run(compressed_efms, infofile, outputfile):
    """
    Entry point for decompressing. Reads compression information form infofile and builds a mapping for deompressions
    which is in reversed order of the previously applied compressions.  Iteratively decompresses one efm after another
    by applying decompressions according to decompression information stored in mappings. If model had additional
    boundaries: removes lambda vector entry and additional boundary reactions from current efm and normalizes it.
    Writes final decompressed efm to user specified output file.

    :param infofile: file automatically created during compressions, containing all information for decompressions
    :param list compressed_efms: compressed efms
    :param compression_infos: info file created during preprocessing
    :param outputfile: path to output file
    :return: None
    """
    counter, bounds = find_counter(infofile)
    mappings = build_reverse_mapping(infofile, counter)
    print("Start decompressions")
    decompressing(compressed_efms, outputfile, mappings, bounds)
