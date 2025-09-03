from fractions import Fraction


def parse_info(file):
    """
    Reads decompression information from infofile and builds decompression list for one2many compressions.

    :param file: infofile automatically created during compressions, containing all information for decompressions
    :return:
        - iterations[::-1] - list of list of tuples containing information on merged reactions and merge factor
        - post - int equal to the amount of reactions before compressions
        - pre - int equal to the amount of reactions after compressions
    """
    iterations = []
    tmp = []
    post = 0
    pre = 0
    for line in file:

        if line.startswith("merged"):
            tokens = line.strip().split()
            post, pre = tokens[1].split(":")

        if line.startswith("remove"):
            infos = []

            for i in line.strip().split():
                if i == "remove" or i == "keep":
                    continue
                tokens = i.split(":")
                if len(tokens) <= 1:
                    continue
                index = int(tokens[0][1:])
                factor = Fraction(tokens[1])
                infos.append((index, factor))
            iterations.append(infos)
        if line.startswith("end"):
            break

    tmp.append(iterations[::-1])
    tmp.append(int(post))
    tmp.append(int(pre))
    return iterations[::-1], int(post), int(pre)


def build_merge_mapping(iterations, post):
    """
    Builds dictionary containing index and merge factor for all reactions.

    :param list iterations: list of list of tuples containing information on merged reactions and merge factor
    :param int post: int equal to the amount of reactions before compressions
    :return: mapping: dictionary containing index and merge factor for all reactions
    """
    insertions = []
    for i in iterations:
        index = i[0][0]
        insertions.append(index)
    insertions = sorted(insertions)
    mapping = {}
    count = 0
    for i in range(0, post):
        if i in insertions:
            continue
        mapping[count] = i
        count += 1
    return mapping


def decompressions(cmp_efm, mapping, iterations, post):
    """
    Decompresses the part of the current compressed efm that has been compressed during one2many compression. This is
    done by splitting reactions that have been merged during nullspace into two reactions. Therefore the merged reaction
    is divided through the merge factor that has been used during one2many compressions.

    :param list cmp_efm: list containing efm for decompression
    :param dict mapping: dictionary containing index and merge factor for all reactions
    :param list iterations: list of list of tuples containing information on merged reactions and merge factor
    :param int post: int equal to the amount of reactions before compressions
    :return: decmp_efm list containing current efm with decompressed one2many part
    """
    decmp_efm = [None] * post
    for i in range(0, len(cmp_efm)):
        decmp_efm[mapping[i]] = cmp_efm[i]

    for it in iterations:
        merged_index, merged_factor = it[0]
        total = 0

        for index, factor in it[1:]:
            total += decmp_efm[index]
        decmp_efm[merged_index] = total / merged_factor

        for index, factor in it[1:]:
            decmp_efm[index] /= factor

    return decmp_efm
