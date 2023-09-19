import pickle
import pandas as pd
import csv
import re
from collections import defaultdict
# from get_training_data import get_non_ascii_lines
from tqdm import tqdm
import numpy as np


def get_csv_as_list(fpath):
    with open(fpath, 'r') as read_obj:
        # Return a reader object which will
        # iterate over lines in the given csvfile
        csv_reader = csv.reader(read_obj)

        # convert string to list
        list_of_csv = list(csv_reader)

    return list_of_csv


def get_lexicon(lex_fpath):
    lex = defaultdict(list)
    with open(lex_fpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pat = re.compile(r'^\("(.+)" (\(?[\w\s\|\-\$]*\)?) (\(.+\))$')
            m = pat.match(line)
            if m:
                word, pos, pron = m.group(1), m.group(2), m.group(3)
                lex[word].append((pos, pron))
    return lex


def get_lex_homograph_prons_dict(lex_fpath, WHD_df):
    with open(WHD_df, "rb") as df:
        WHD = pickle.load(df)

    homographs = WHD['homograph'].tolist()
    homograph_set = list(set(homographs))

    # get lexicon in form 'winds': [('(nns air)', '(((w i n d z) 1)))'), ('(vbz air)', '(((w i n d z) 1)))'), ('(vbz turn)', '(((w ai n d z) 1)))')]
    lexicon = get_lexicon(lex_fpath)

    # get just the homograph entries in lexicon (len = 162)
    homograph_lexicon = {homograph: lexicon[homograph] for homograph in homograph_set}

    # get the homograph entries in the format 'winds': [('nns air', 'w i n d z', [1])]
    # 'correlate': [('nn', 'k o - r @ - l @ t', [1, 0, 0,])]
    # delete brackets, replace digits with -, delete trailing -, add list of digits as ints as third element of tuple

    homograph_lex_processed = defaultdict(list)
    for homograph, prons in homograph_lexicon.items():
        new_value = []
        for pron in prons:
            processed_pos = pron[0].replace("(", "").replace(")", "").rstrip()
            stresses = [int(digit) for digit in re.findall(r"\d", pron[1])]
            processed_pron = pron[1].replace("(", "").replace(")", "").replace("0", "-").replace("1", "-").replace("2", "-").replace("3", "-").strip("-").rstrip()
            new_value.append((processed_pos, processed_pron, stresses))
        homograph_lex_processed[homograph] = new_value

    return homograph_lex_processed


def edit_WHD_df(WHD_df_fname):
    with open(WHD_df_fname, "rb") as infile:
        WHD = pickle.load(infile)

    set_word_ids = ["bass", "produce_nou", "discard_vrb", "contract_vrb", "confines_nou", "escort_vrb", "ornament_vrb", \
                    "approximate_adj-nou", "misuse_nou", "pasty_nou", "export_vrb", "invalid_adj", "compress", "live_adj", \
                    "deliberate_adj", "perfect_vrb", "isolate", "perfume_nou", "exploit_nou", "extract_nou", "ravel_nam", \
                    "compress_nou", "postulate_vrb", "combine_nou", "articulate_vrb", "addict_nou", "refund_vrb", "house_vrb", \
                    "implant_nou", "nestle_vrb", "pervert_vrb", "project_vrb", "extract_vrb", "instrument_nou", "converse_vrb", \
                    "implement_nou", "close_vrb", "advocate_nou", "lives_nou", "impact_vrb", "separate_vrb", "contrast_vrb", \
                    "sow_nou", "perfect_adj", "graduate_adj-nou", "exploit_vrb", "insert_vrb", "pasty_adj", "read_past", "recount_nou", \
                    "record_nou", "rerelease_vrb", "discount_nou", "contrast_nou", "diffuse_adj", "syndicate_nou", "compound_vrb", \
                    "wind_nou", "job", "discount_vrb", "august_nam", "celtic_adj-nou-sports", "protest_vrb", "deviate_vrb", "suspect_adj-nou", \
                    "alternate_adj-nou", "appropriate_adj", "degenerate_adj-nou", "consort_nou", "conglomerate_adj-nou", \
                    "separate_adj", "subject_adj-nou", "polish", "celtic", "progress_vrb", "incline_vrb", "expose_nou", "alternate_vrb", \
                    "coordinate_vrb", "content_nou", "moderate_vrb", "jesus", "increase_vrb", "affect", "reading_en", "aggregate_adj-nou", \
                    "discharge_nou", "compound_nou", "postulate_nou", "desert_nou", "blessed_vrb", "coordinate_adj-nou", "interchange_nou", \
                    "mouth_nou", "escort_nou", "deviate_nou", "rodeo", "nestle_nam", "conflict_vrb", "uses_vrb", "aged_adj", \
                    "abstract_adj-nou", "construct_nou", "mobile", "retard_nou", "lead_nou-vrb", "subordinate_adj-nou", "precipitate_adj-nou", \
                    "construct_vrb", "august", "upset_vrb", "ornament_nou", "rebel_vrb", "combine_vrb", "transport_nou", \
                    "diagnoses_nou", "suspect_vrb", "refuse_vrb", "uses_nou", "pervert_nou", "converse_adj-nou", "invert_adj-nou", \
                    "advocate_vrb", "overthrow_nou", "increase_nou", "correlate_nou", "predicate_vrb", "recount_vrb", "invite_vrb", \
                    "abuses_nou", "transplant_nou", "diagnoses_vrb", "insert_nou", "resume_vrb", "winds_vrb", "use_vrb", "entrance_nou", "bologna", \
                    "jesus_es", "moderate_adj-nou", "abstract_vrb", "invalid_nou", "buffet_vrb", "axes_nou-vrb", "delegate_nou", "content_adj-nou-vrb", \
                    "delegate_vrb", "excuse_vrb", "abuse_vrb", "convert_nou", "resume_nou", "syndicate_vrb", "expatriate_nou", "expose_vrb", \
                    "sake_jp", "insult_nou", "invert_vrb", "lead_nou", "winds_nou", "tear_vrb", "conflict_nou", "convict_vrb", "wound_vrb", \
                    "rodeo_geo", "axes_nou", "tear_nou", "diffuse_vrb", "lives_vrb", "abuses_vrb", "contract_nou", "learned_vrb", "conjugate_adj-nou", \
                    "aged", "degenerate_vrb", "implement_vrb", "deliberate_vrb", "pigment_nou", "refund_nou", "associate_adj-nou", \
                    "initiate_nou", "live_vrb", "convert_vrb", "console_nou", "minute_adj", "convict_nou", "frequent_adj", "moped_nou", \
                    "confines_vrb", "wound_nou-vrb", "estimate_vrb", "articulate_adj", "impact_nou", "protest_nou", "object_nou", \
                    "transform", "produce_vrb", "retard_vrb", "row_1", "conjugate_vrb", "isolate_nou", "transform_nou", "learned_adj", \
                    "conglomerate_vrb", "rebel_nou", "present_adj-nou", "implant_vrb", "read_present", "misuse_vrb", "upset_nou", \
                    "bologna_geo", "bow_nou-knot", "affiliate_nou", "discard_nou", "subordinate_vrb", "permit_vrb", "perfume_vrb", \
                    "document_vrb", "transplant_vrb", "consummate_adj", "dove", "refuse_nou", "mate", "frequent_vrb", "precipitate_vrb", \
                    "use_nou", "defect_nou", "duplicate_adj-nou", "export_nou", "conscript_nou", "polish_geo", "reject_nou", "excuse_nou", \
                    "analyses_vrb", "ravel_nou", "bow_nou-ship", "correlate_nou-vrb", "predicate_nou", "supplement_vrb", "document_nou", \
                    "expatriate_vrb", "import_vrb", "affect_nou-psy", "fragment_nou", "reject_vrb", "intrigue_nou", "laminate_nou", \
                    "attribute_vrb", "bass_corp", "contest_vrb", "abuse_nou", "insult_vrb", "conduct_vrb", "laminate_vrb", \
                    "wind_vrb", "aggregate_vrb", "compact_adj-nou", "import_nou", "decrease_nou", "close_adj-nou", "elaborate_adj", \
                    "sow", "increment_vrb", "mate_nou", "dove_vrb", "appropriate_vrb", "august_adj", "graduate_vrb", "conduct_nou", \
                    "transport_vrb", "rerelease", "decrease_vrb", "duplicate_vrb", "discharge_vrb", "analyses_nou", "sake", \
                    "increment_nou", "mobile_geo", "reading_geo", "defect_vrb", "house_nou", "invite_nou", "estimate_nou", \
                    "intrigue_nou-vrb", "permit_nou", "approximate_vrb", "project_nou", "attribute_nou", "animate_adj-nou", \
                    "record_vrb", "minute", "overthrow_vrb", "animate_vrb", "affiliate_vrb", "initiate_vrb", "contest_nou", \
                    "supplement_nou", "conscript_vrb", "progress_nou", "associate_vrb", "intimate_adj", "incline_nou", "incense_nou", \
                    "console_vrb", "buffet_nou", "present_vrb", "consummate_vrb", "blessed_adj", "fragment_vrb"]
    gt_pronunciations = [('nn musical', 'b ei s', [1]), ('nn', 'p r o - jh uu s', [1, 0]), ('vb', 'd i - s k aa d', [0, 1]),
             ('vb', 'k @ n - t r a k t', [0, 1]), ('nns', 'k o n - f ai n z', [1, 0]), ('vb', 'i - s k oo t', [0, 1]),
             ('vb', 'oo - n @ - m e n t', [1, 0, 0]), ('jj', '@ - p r o k - s i - m @ t', [0, 1, 0, 0]),
             ('nn', 'm i s - y uu s', [0, 1]), ('nn', 'p a - s t iy', [1, 0]), ('vb', 'i k - s p oo t', [0, 1]),
             ('jj', 'i n - v a - l i d', [0, 1, 0]), ('vb', 'k @ m - p r e s', [0, 1]), ('jj', 'l ai v', [1]),
             ('jj', 'd i - l i - b @ - r @ t', [0, 1, 0, 0]), ('vb', 'p @ - f e k t', [0, 1]),
             ('vb', 'ai - s @ - l ei t', [1, 0, 0]), ('nn', 'p @@r - f y uu m', [1, 0]),
             ('nn', 'e k - s p l oi t', [1, 0]),
             ('nn', 'e k - s t r a k t', [1, 0]), ('nnp', 'r @ - v e lw', [0, 1]), ('nn', 'k o m - p r e s', [1, 0]),
             ('vb', 'p o s - ch uu - l ei t', [1, 0, 0]), ('nn', 'k o m - b ai n', [1, 0]),
             ('vb', 'aa - t i - k y uw - l ei t', [0, 1, 0, 0]), ('nn', 'a - d i k t', [1, 0]),
             ('vb', 'r i - f uh n d', [0, 1]), ('vb', 'h ow z', [1]), ('nn', 'i m - p l aa n t', [1, 0]),
             ('vb', 'n e - s l!', [1, 0]), ('vb', 'p @ - v @@r t', [0, 1]), ('vb', 'p r @ - jh e k t', [0, 1]),
             ('vb', 'i k - s t r a k t', [0, 1]), ('nn', 'i n - s t r @ - m @ n t', [1, 0, 0]),
             ('vb', 'k @ n - v @@r s', [0, 1]), ('nn', 'i m - p l i - m @ n t', [1, 0, 0]), ('vb', 'k l ou z', [1]),
             ('nn', 'a d - v @ - k @ t', [1, 0, 0]), ('nns', 'l ai v z', [1]), ('vb', 'i m - p a k t', [0, 1]),
             ('vb', 's e - p @ - r ei t', [1, 0, 0]), ('vb', 'k @ n - t r aa s t', [0, 1]), ('nn', 's ow', [1]),
             ('jj', 'p @@r - f i k t', [1, 0]), ('jj', 'g r a - jh uw @ t', [1, 0]), ('vb', 'i k - s p l oi t', [0, 1]),
             ('vb', 'i n - s @@r t', [0, 1]), ('jj', 'p ei s t - iy', [1, 0]), ('vbd', 'r e d', [1]),
             ('nn', 'r ii - k ow n t', [1, 0]), ('nn', 'r e - k oo d', [1, 0]),
             ('vb', 'r ii - r i - l ii s', [2, 0, 1]),
             ('nn reduce', 'd i - s k ow n t', [1, 0]), ('nn', 'k o n - t r aa s t', [1, 0]),
             ('jj', 'd i - f y uu s', [0, 1]),
             ('nn', 's i n - d i - k @ t', [1, 0, 0]), ('vb', 'k @ m - p ow n d', [0, 1]), ('nn air', 'w i n d', [1]),
             ('nn', 'jh o b', [1]), ('vb reduce', 'd i - s k ow n t', [1, 0]), ('nnp', 'oo - g @ s t', [1, 0]),
             ('nnp', 's e lw - t i k', [1, 0]), ('vb', 'p r @ - t e s t', [0, 1]),
             ('vb', 'd ii - v iy - ei t', [1, 0, 0]),
             ('nn', 's uh - s p e k t', [1, 0]), ('jj', 'oo lw - t @@r - n @ t', [0, 1, 0]),
             ('jj', '@ - p r ou - p r iy @ t', [0, 1, 0]), ('jj', 'd i - jh e - n @ - r @ t', [0, 1, 0, 0]),
             ('nn', 'k o n - s oo t', [1, 0]), ('nn', 'k @ ng - g l o - m @ - r @ t', [0, 1, 0, 0]),
             ('jj', 's e - p @ - r @ t', [1, 0, 0]), ('jj', 's uh b - jh i k t', [1, 0]),
             ('nn', 'p o - l i sh', [1, 0]),
             ('jj', 'k e lw t - i k', [1, 0]), ('vb', 'p r @ - g r e s', [0, 1]), ('vb', 'i n - k l ai n', [0, 1]),
             ('nn', 'e k - s p ou - z ei', [0, 1, 0]), ('vb', 'oo lw - t @ - n ei t', [1, 0, 0]),
             ('vb', 'k ou - oo - d i n - ei t', [0, 1, 0, 0]), ('nn', 'k o n - t e n t', [1, 0]),
             ('vb', 'm o - d @ - r ei t', [1, 0, 0]), ('nnp', 'jh ii - z @ s', [1, 0]),
             ('vb', 'i n - k r ii s', [0, 1]),
             ('vb', '@ - f e k t', [0, 1]), ('nn', 'r ii d - i ng', [1, 0]), ('jj', 'a - g r i - g @ t', [1, 0, 0]),
             ('nn', 'd i s - ch aa jh', [1, 0]), ('nn', 'k o m - p ow n d', [1, 0]),
             ('nn', 'p o s - ch uu - l @ t', [1, 0, 0]),
             ('nn', 'd e - z @ t', [1, 0]), ('vbd', 'b l e s t', [1]), ('nn', 'k ou - oo - d i n - @ t', [0, 1, 0, 0]),
             ('nn', 'i n - t @ - ch ei n jh', [1, 0, 2]), ('nn', 'm ow th', [1]), ('nn', 'e - s k oo t', [1, 0]),
             ('nn', 'd ii - v iy @ t', [1, 0]), ('nn', 'r ou - d iy - ou', [1, 0, 0]), ('nnp', 'n e s - l ei', [1, 0]),
             ('vb', 'k @ n - f l i k t', [0, 1]), ('vbz', 'y uu z - i z', [1, 0]), ('jj old', 'ei jh - i d', [1, 0]),
             ('jj summarise', 'a b - s t r a k t', [1, 0]), ('nn', 'k o n - s t r uh k t', [1, 0]),
             ('jj', 'm ou - b ai lw', [1, 0]), ('nn', 'r ii - t aa d', [1, 0]), ('nn non-metal', 'l ii d', [1]),
             ('jj', 's @ - b oo - d i n - @ t', [0, 1, 0, 0]), ('jj', 'p r i - s i - p i - t @ t', [0, 1, 0, 0]),
             ('vb', 'k @ n - s t r uh k t', [0, 1]), ('nnp', 'oo - g @ s t', [1, 0]), ('vb', 'uh p - s e t', [3, 1]),
             ('nn', 'oo - n @ - m @ n t', [1, 0, 0]), ('vb', 'r i - b e lw', [0, 1]), ('vb', 'k @ m - b ai n', [0, 1]),
             ('nn', 't r a n - s p oo t', [1, 0]), ('nns', 'd ai @ g - n ou - s ii z', [0, 1, 0]),
             ('vb', 's @ - s p e k t', [0, 1]), ('vb', 'r i - f y uu z', [0, 1]), ('nns', 'y uu s - i z', [1, 0]),
             ('nn', 'p @@r - v @@r t', [1, 0]), ('nn', 'k o n - v @@r s', [1, 0]), ('jj', 'i n - v @@r t', [1, 0]),
             ('vb', 'a d - v @ - k ei t', [1, 0, 0]), ('nn', 'ou - v @ - th r ou', [1, 0, 2]),
             ('nn', 'i n - k r ii s', [1, 0]),
             ('nn', 'k o - r @ - l @ t', [1, 0, 0]), ('vb', 'p r e - d i - k ei t', [1, 0, 0]),
             ('vb tell', 'r i - k ow n t', [0, 1]), ('vb', 'i n - v ai t', [0, 1]),
             ('nns', '@ - b y uu s - i z', [0, 1, 0]),
             ('nn', 't r a n s - p l aa n t', [1, 0]), ('vbz', 'd ai @ g - n ou z - i z', [1, 0, 0]),
             ('nn', 'i n - s @@r t', [1, 0]), ('vb', 'r i - z y uu m', [0, 1]), ('vbz turn', 'w ai n d z', [1]),
             ('vb', 'y uu z', [1]), ('nn', 'e n - t r @ n s', [1, 0]), ('nn', 'b @ - l ou - n iy', [0, 1, 0]),
             ('nnp', 'jh ii - z @ s', [1, 0]), ('jj', 'm o - d @ - r @ t', [1, 0, 0]),
             ('vb remove', '@ b - s t r a k t', [0, 1]), ('nn', 'i n - v @ - l i d', [1, 0, 0]),
             ('vb sideboard', 'b uh - f i t', [1, 0]), ('nns plural-of-axe', 'a k s - i z', [1, 0]),
             ('nn', 'd e - l i - g @ t', [1, 0, 0]), ('jj', 'k @ n - t e n t', [0, 1]),
             ('vb', 'd e - l i - g ei t', [1, 0, 0]),
             ('vb', 'i k - s k y uu z', [0, 1]), ('vb', '@ - b y uu z', [0, 1]), ('nn', 'k o n - v @@r t', [1, 0]),
             ('nn', 'r e z - y uw - m ei', [1, 0, 0]), ('vb', 's i n - d i - k ei t', [1, 0, 0]),
             ('nn', 'e k - s p a - t r iy @ t', [0, 1, 0]), ('vb', 'i k - s p ou z', [0, 1]),
             ('fw drink', 's aa - k ei', [1, 0]), ('nn', 'i n - s uh lw t', [1, 0]), ('vb', 'i n - v @@r t', [0, 1]),
             ('nn metal', 'l e d', [1]), ('nns air', 'w i n d z', [1]), ('vb rip', 't eir', [1]),
             ('nn', 'k o n - f l i k t', [1, 0]), ('vb', 'k @ n - v i k t', [0, 1]), ('vbd', 'w ow n d', [1]),
             ('nnp', 'r ou - d iy - ou', [1, 0, 0]), ('nns plural-of-axis', 'a k - s ii z', [1, 0]),
             ('nn water', 't i@', [1]),
             ('vb', 'd i - f y uu z', [0, 1]), ('vbz', 'l i v z', [1]), ('vbz', '@ - b y uu z - i z', [0, 1, 0]),
             ('nn', 'k o n - t r a k t', [1, 0]), ('vbd', 'l @@r n d', [1]), ('nn', 'k o n - jh u - g @ t', [1, 0, 0]),
             ('jj of-age', 'ei jh d', [1]), ('vb', 'd i - jh e - n @ - r ei t', [0, 1, 0, 0]),
             ('vb', 'i m - p l i - m e n t', [1, 0, 0]), ('vb', 'd i - l i - b @ - r ei t', [0, 1, 0, 0]),
             ('nn', 'p i g - m @ n t', [1, 0]), ('nn', 'r ii - f uh n d', [1, 0]),
             ('jj', '@ - s ou - s iy @ t', [0, 1, 0]),
             ('nn', 'i - n i - sh iy @ t', [0, 1, 0]), ('vb', 'l i v', [1]), ('vb', 'k @ n - v @@r t', [0, 1]),
             ('nn', 'k o n - s ou lw', [1, 0]), ('jj', 'm ai - n y uu t', [0, 1]), ('nn', 'k o n - v i k t', [1, 0]),
             ('jj', 'f r ii - k w @ n t', [1, 0]), ('nn', 'm ou - p e d', [1, 0]), ('vbz', 'k @ n - f ai n z', [0, 1]),
             ('nn', 'w uu n d', [1]), ('vb', 'e - s t i - m ei t', [1, 0, 0]),
             ('jj', 'aa - t i - k y uw - l @ t', [0, 1, 0, 0]), ('nn', 'i m - p a k t', [1, 0]),
             ('nn', 'p r ou - t e s t', [1, 0]), ('nn', 'o b - jh i k t', [1, 0]), ('vb', 't r a n s - f oo m', [0, 1]),
             ('vb', 'p r @ - d y uu s', [0, 1]), ('vb', 'r i - t aa d', [0, 1]), ('vb boating', 'r ou', [1]),
             ('vb', 'k o n - jh u - g ei t', [1, 0, 0]), ('vbp', 'ai - s @ - l ei t', [1, 0, 0]),
             ('vb', 't r a n s - f oo m', [0, 1]), ('jj', 'l @@r n - i d', [1, 0]),
             ('vb', 'k @ ng - g l o - m @ - r ei t', [0, 1, 0, 0]), ('nn', 'r e - b l!', [1, 0]),
             ('jj', 'p r e - z n! t', [1, 0]), ('vb', 'i m - p l aa n t', [0, 1]), ('vbp', 'r ii d', [1]),
             ('vb', 'm i s - y uu z', [0, 1]), ('jj', 'uh p - s e t', [3, 1]), ('nnp', 'b @ - l ou - n y @', [0, 1, 0]),
             ('nn', 'b ou', [1]), ('nn', '@ - f i - l iy @ t', [0, 1, 0]), ('nn', 'd i - s k aa d', [1, 0]),
             ('vb', 's @ - b oo - d i n - ei t', [0, 1, 0, 0]), ('vb', 'p @ - m i t', [0, 1]),
             ('vb', 'p @ - f y uu m', [0, 1]),
             ('vb', 'd o - k y uw - m e n t', [1, 0, 0]), ('vb', 't r a n s - p l aa n t', [0, 1]),
             ('jj', 'k o n - s @ - m @ t', [1, 0, 0]), ('nn', 'd uh v', [1]), ('nn', 'r e f - y uu s', [1, 0]),
             ('nn', 'm ei t', [1]), ('vb', 'f r i - k w e n t', [0, 1]),
             ('vb', 'p r i - s i - p i - t ei t', [0, 1, 0, 0]),
             ('nn', 'y uu s', [1]), ('nn', 'd ii - f e k t', [1, 0]), ('jj', 'd y uu - p l i - k @ t', [1, 0, 0]),
             ('nn', 'e k - s p oo t', [1, 0]), ('nn', 'k o n - s k r i p t', [1, 0]), ('nnps', 'p ou l - i sh', [1, 0]),
             ('nn', 'r ii - jh e k t', [1, 0]), ('nn', 'i k - s k y uu s', [0, 1]),
             ('vbz', 'a - n @ l - ai z - i z', [1, 0, 2, 0]), ('vb', 'r a - v l!', [1, 0]), ('vb', 'b ow', [1]),
             ('vb', 'k o - r @ - l ei t', [1, 0, 0]), ('nn', 'p r e - d i - k @ t', [1, 0, 0]),
             ('vb', 's uh - p l i - m e n t', [1, 0, 0]), ('nn', 'd o - k y uw - m @ n t', [1, 0, 0]),
             ('vb', 'e k - s p a - t r iy - ei t', [0, 1, 0, 0]), ('vb', 'i m - p oo t', [0, 1]),
             ('vb', '@ - f e k t', [0, 1]),
             ('nn', 'f r a g - m @ n t', [1, 0]), ('vb', 'r i - jh e k t', [0, 1]), ('nn', 'i n - t r ii g', [1, 0]),
             ('nn', 'l a - m i - n ei t', [1, 0, 0]), ('vb', '@ - t r i - b y uu t', [0, 1, 0]),
             ('nn fish', 'b a s', [1]),
             ('vb', 'k @ n - t e s t', [0, 1]), ('nn', '@ - b y uu s', [0, 1]), ('vb', 'i n - s uh lw t', [0, 1]),
             ('vb', 'k @ n - d uh k t', [0, 1]), ('jj', 'l a - m i - n ei t', [1, 0, 0]), ('vb turn', 'w ai n d', [1]),
             ('vb', 'a - g r i - g ei t', [1, 0, 0]), ('jj', 'k o m - p a k t', [1, 0]), ('nn', 'i m - p oo t', [1, 0]),
             ('nn', 'd ii - k r ii s', [1, 0]), ('jj', 'k l ou s', [1]), ('jj', 'i - l a - b @ - r @ t', [0, 1, 0, 0]),
             ('vb', 's ou', [1]), ('vb', 'i n - k r @ - m e n t', [1, 0, 0]), ('nn', 'm ei t', [1]),
             ('vbd', 'd ou v', [1]),
             ('vb', '@ - p r ou - p r iy - ei t', [0, 1, 0, 0]), ('jj', 'oo - g uh s t', [0, 1]),
             ('vb', 'g r a - jh uw - ei t', [1, 0, 0]), ('nn', 'k o n - d uh k t', [1, 0]),
             ('vb', 't r a n - s p oo t', [0, 1]), ('nn', 'r ii - r i - l ii s', [1, 0, 2]),
             ('vb', 'd i - k r ii s', [0, 1]),
             ('vb', 'd y uu - p l i - k ei t', [1, 0, 0]), ('vb', 'd i s - ch aa jh', [0, 1]),
             ('nns', '@ - n a - l @ - s ii z', [0, 1, 0, 0]), ('nn benefit', 's ei k', [1]),
             ('nn', 'i n - k r @ - m @ n t', [1, 0, 0]), ('jj', 'm ou - b ai lw', [1, 0]),
             ('nnp', 'r e - d i ng', [1, 0]),
             ('vb', 'd i - f e k t', [0, 1]), ('nn', 'h ow s', [1]), ('nn', 'i n - v ai t', [1, 0]),
             ('nn', 'e - s t i - m @ t', [1, 0, 0]), ('vb', 'i n - t r ii g', [0, 1]), ('nn', 'p @@r - m i t', [1, 0]),
             ('vb', '@ - p r o k - s i - m ei t', [0, 1, 0, 0]), ('nn', 'p r o - jh e k t', [1, 0]),
             ('nn', 'a - t r i - b y uu t', [1, 0, 2]), ('jj', 'a - n i - m @ t', [1, 0, 0]),
             ('vb', 'r i - k oo d', [0, 1]),
             ('nn', 'm i - n i t', [1, 0]), ('vb', 'ou - v @ - th r ou', [2, 0, 1]),
             ('vb', 'a - n i - m ei t', [1, 0, 0]),
             ('vb', '@ - f i - l iy - ei t', [0, 1, 0, 0]), ('vb', 'i - n i - sh iy - ei t', [0, 1, 0, 0]),
             ('nn', 'k o n - t e s t', [1, 0]), ('nn', 's uh - p l i - m @ n t', [1, 0, 0]),
             ('vb', 'k @ n - s k r i p t', [0, 1]), ('nn', 'p r ou - g r e s', [1, 0]),
             ('vb', '@ - s ou - s iy - ei t', [0, 1, 0, 0]), ('jj', 'i n - t i - m @ t', [1, 0, 0]),
             ('nn', 'i n - k l ai n', [1, 0]), ('nn', 'i n - s e n s', [1, 0]), ('vb', 'k @ n - s ou lw', [0, 1]),
             ('nn dining-car', 'b u - f ei', [1, 0]), ('vb', 'p r i - z e n t', [0, 1]),
             ('vb', 'k o n - s @ - m ei t', [1, 0, 0]), ('jj', 'b l e s - i d', [1, 0]),
             ('vb', 'f r a g - m e n t', [0, 1])]

    zipped_id_pron = list(zip(set_word_ids, gt_pronunciations))

    df = pd.DataFrame(zipped_id_pron, columns=['wordid', 'gt_homograph_pron'])

    with open("GT_prons_ids.pkl", "wb") as outfile:
        pickle.dump(df, outfile)

    WHD = WHD.merge(df, on='wordid', how='left')

    WHD['SpaCy_POSseq'] = get_csv_as_list("seqs/pos_seqs_WHD")
    WHD['SpaCy_tokens'] = get_csv_as_list("seqs/token_seqs_WHD")
    WHD['words'] = get_csv_as_list("seqs/word_seqs_WHD")

    WHD = WHD.drop("ElementMatch", axis=1)
    WHD = WHD.drop("spacy_POS", axis=1)
    WHD = WHD.drop("POS_label", axis=1)
    WHD = WHD.drop("start", axis=1)
    WHD = WHD.drop("end", axis=1)

    print(WHD.head().to_string())
    print(WHD.info())
    input()

    with open("WHD_full.pkl", "wb") as outfile:
        pickle.dump(WHD, outfile)


def get_gt_correct_format(df, fpath):
    with open(df, "rb") as infile:
        WHD = pickle.load(infile)

    gt_prons = WHD['gt_homograph_pron'].tolist()
    homographs = WHD['homograph'].tolist()

    processed_gt_prons = []
    for gt_pron in gt_prons:
        pos, pron, stresses = gt_pron
        split_pron = pron.split(" - ")
        restored_pron = ""
        for stress, syl in zip(stresses, split_pron):
            restored_pron += f"{stress}"
            restored_pron += f" {syl}"
            restored_pron += " - "
        processed_gt_prons.append(restored_pron[:-3])

    with open(fpath, 'w') as f:
        for pron in processed_gt_prons:
            f.write(pron + "\n")

    # with open("clean_eval_homographs_NEW.txt", "w") as f:
    #     for homograph in homographs:
    #         f.write(homograph + "\n")


# get_gt_correct_format("WHD_eval_clean_NEW.pkl", "clean_eval_gt_prons_NEW.txt")


def get_homograph_pron_stats(phonetic_seqs):
    """
    Get statistics for homograph prons from festival phonetic seqs as compared to gt_homograph_prons and write fully_correct,
    pron_correct and pron_incorrect predictions each to a csv.

    Args:
    phonetic_seqs = file of Festival phonetic output for WHD
    df = WHD data as a df with word_seqs, token_seqs, pos_seqs, SpaCy homograph pos pred

    Return:
    list of indices for preds that need correction
    """

    with open(phonetic_seqs, "r") as f:
        preds = f.readlines()

    with open(df, "rb") as f:
        WHD = pickle.load(f)

    wordids = WHD['wordid'].tolist()


    # assign correct pron in gt_homograph_pron to wordids that do not have a gt homograph pron in the unilex-rpx lexicon - DONE
    # missing_pron_wordids = ['isolate_nou', 'affect_nou_psy', 'jesus_es', 'rodeo_geo', 'transform_nou', 'laminate_nou', 'mate_nou', 'mobile_geo']
    # missing_prons = [("nn", "ai - s @ - l @ t", [1,0,0]), ("nn", "a - f e k t", [1,0]), ("nnp", "h ei - z uu s", [0, 1]), ("nnp", "r ou - d ei - ou", [0, 1, 0]), ("nn", "t r a n s - f oo m", [1, 0]), ("nn", "l a - m i - n @ t", [1, 0, 0]), ("nn", "m a - t ei", [1, 0]), ("nnp", "m ou - b ii lw", [0,1])]
    #
    # for i, (row, wid) in enumerate(zip(WHD.iterrows(), wordids)):
    #     if wid in missing_pron_wordids:
    #         idx = missing_pron_wordids.index(wid)
    #         missing_pron = missing_prons[idx]
    #         WHD.at[i, 'gt_homograph_pron'] = missing_pron


    # split the pron sequences into list of prons for each word in each sentence in the form '0 th @@r - 1 t ii n th'
    all_preds = []
    for pred in preds:
        pred_prons = pred.rstrip().replace('_B', '+').split('+')[:-1]  # remove final empty string
        # remove the leading and trailing blank spaces
        pred_prons = list(map(str.strip, pred_prons))
        all_preds.append(pred_prons)

    # process the predicted pronunciations so that they are in the form ('th @@r - t ii n th', [0, 1])
    all_processed_preds = []
    for pred in all_preds:
        processed_preds = []
        for pron in pred:
            stresses = [int(digit) for digit in re.findall(r"\d", pron)]
            processed_pron = pron.replace("0 ", "").replace("1 ", "").replace("2 ", "").replace("3 ", "")
            processed_preds.append((processed_pron, stresses))
        all_processed_preds.append(processed_preds)


    gt_homograph_prons = WHD['gt_homograph_pron'].tolist()

    fully_correct = 0
    just_pronunciation_correct = 0
    fully_incorrect = 0
    not_fully_correct = 0
    bad_preds = []
    correct_prons = []
    correct_pron_stress = []

    for i, (pred, gt) in enumerate(zip(all_processed_preds, gt_homograph_prons)):
        if gt:
            gt_pron = gt[1]
            gt_stresses = gt[2]
            gt_pron_stresses = (gt_pron, gt_stresses)
            try:
                pred_pron_stress_idx = pred.index(gt_pron_stresses)
                correct_pron_stress.append([i, pred, gt_pron_stresses])
                fully_correct += 1
            except ValueError:
                not_fully_correct += 1
                found = False
                for word_pred in pred:
                    if word_pred[0] == gt_pron:
                        just_pronunciation_correct += 1
                        correct_prons.append([i, pred, gt_pron_stresses])
                        found = True
                        break
                if found is False:
                    fully_incorrect += 1
                    bad_preds.append([i, pred, gt_pron_stresses])

    print("\nfully correct: ", fully_correct)
    print("not fully correct: ", not_fully_correct)
    print("fully + not fully correct: ", fully_correct + not_fully_correct)
    print("")
    print("just pron correct: ", just_pronunciation_correct)
    print("pronunciation incorrect: ", fully_incorrect)
    print("fully correct + fully incorrect + just pron + partial pron: ", just_pronunciation_correct + fully_incorrect + fully_correct, "\n")

    # with open("WHD_seq2seq_analysis/FE_POS/pronunciation_incorrect.csv", 'w') as f:
    #     # using csv.writer method from CSV package
    #     write = csv.writer(f)
    #     write.writerows(bad_preds)
    #
    # with open("WHD_seq2seq_analysis/FE_POS/just_pronunciation_correct.csv", 'w') as f:
    #     # using csv.writer method from CSV package
    #     write = csv.writer(f)
    #     write.writerows(correct_prons)
    #
    # with open("WHD_seq2seq_analysis/FE_POS/fully_correct.csv", "w") as f:
    #     # using csv.writer method from CSV package
    #     write = csv.writer(f)
    #     write.writerows(correct_pron_stress)

    bad_idxs = []
    correct_pron_idxs = []

    for bad_line in bad_preds:
        bad_idx = bad_line[0]
        bad_idxs.append(bad_idx)

    for correct_pron in correct_prons:
        correct_pron_idx = correct_pron[0]
        correct_pron_idxs.append(correct_pron_idx)

    return bad_idxs, correct_pron_idxs


def find_indices(search_list, search_item):
    return [index for (index, item) in enumerate(search_list) if item == search_item]


def correct_preds(phonetic_seqs, df, homograph_dict):
    bad_idxs, only_pron_correct_idxs = get_homograph_pron_stats(phonetic_seqs, df)
    idxs = sorted(bad_idxs + only_pron_correct_idxs)

    with open(df, "rb") as f:
        WHD = pickle.load(f)

    with open(phonetic_seqs, "r") as f:
        seqs = f.readlines()

    gt_homograph_prons = WHD['gt_homograph_pron']
    words = WHD['words']
    homographs = WHD['homograph']

    all_preds = []
    for pred in seqs:
        pred_prons = pred.rstrip().replace('_B', '+').split('+')[:-1]  # remove final empty string
        # remove the leading and trailing blank spaces
        pred_prons = list(map(str.strip, pred_prons))
        all_preds.append(pred_prons)

    # process the predicted pronunciations so that they are in the form ('th @@r - t ii n th', [0, 1])
    all_processed_preds = []
    for pred in all_preds:
        processed_preds = []
        for pron in pred:
            stresses = [int(digit) for digit in re.findall(r"\d", pron)]
            processed_pron = pron.replace("0 ", "").replace("1 ", "").replace("2 ", "").replace("3 ", "")
            processed_preds.append((processed_pron, stresses))
        all_processed_preds.append(processed_preds)

    corrected_preds = []
    for i, (homograph, word_seq, pred_seq, gt) in enumerate(zip(homographs, words, all_processed_preds, gt_homograph_prons)):
        if len(word_seq) == len(pred_seq):
            homograph_ids = find_indices(word_seq, homograph)
            for id in homograph_ids:
                pred_seq[id] = (gt[1], gt[2])
            corrected_preds.append(pred_seq)
        else:
            potential_preds = homograph_dict[homograph]
            for pot_pred in potential_preds:
                pot_pred = (pot_pred[1], pot_pred[2])
                if pot_pred in pred_seq:
                    homograph_ids = find_indices(pred_seq, pot_pred)
                    for id in homograph_ids:
                        pred_seq[id] = (gt[1], gt[2])
                    corrected_preds.append(pred_seq)
                    break

    assert len(corrected_preds) == 16102

    with open("festival_analysis/corrected_preds_full.csv", 'w') as word:
        # using csv.writer method from CSV package
        write = csv.writer(word)
        write.writerows(corrected_preds)

    return corrected_preds


# correct_preds("WHD_data/data/WHD_pron.txt", "WHD_full.pkl", get_lex_homograph_prons_dict("unilex-rpx.out", "WHD_full.pkl"))


def postprocess_preds(corrected_preds, og_phonetic_seqs, out_file):
    with open(og_phonetic_seqs, "r") as f:
        phonetic_seqs = f.readlines()

    restored_phonetic_seqs = []
    for line, seq in zip(corrected_preds, phonetic_seqs):
        seq = seq.strip()
        split_seq = re.split(r"(\+)|(_B)", seq)
        clean_split_seq = [i for i in split_seq if i is not None][:-1] # get rid of surplus None matches
        clean_split_seq = list(map(str.strip, clean_split_seq)) # strip leading and trailing whitespaces from each list element
        break_idxs = find_indices(clean_split_seq, "_B")
        restored_line = []
        for word_tuple in line:
            string, stresses = word_tuple
            split_word = string.split(" - ")
            restored_word = ""
            for stress, syl in zip(stresses, split_word):
                restored_word += f"{stress}"
                restored_word += f" {syl}"
                restored_word += " - "
            restored_line.append(restored_word[:-3])
            restored_line.append("+")
        joined_line = " ".join(restored_line)
        split_line = re.split(r"(\+)|(_B)", joined_line)
        clean_split_line = [i for i in split_line if i is not None][:-1]  # get rid of surplus None matches
        clean_split_line = list(map(str.strip, clean_split_line)) # strip leading and trailing whitespaces from each list element
        for idx in break_idxs:
            clean_split_line[idx] = "_B"
        restored_phonetic_seqs.append(" ".join(clean_split_line))

    with open(out_file, "w") as f:
        for line in restored_phonetic_seqs:
            f.write(line + "\n")


# postprocess_preds(correct_preds("WHD_data/data/WHD_pron.txt", "WHD_full.pkl", get_lex_homograph_prons_dict("unilex-rpx.out", "WHD_full.pkl"))
# , "WHD_data/data/WHD_pron.txt", "WHD_data/data/WHD_tgt.txt")


def get_gt_homograph_prons(homograph_lex):
    # with open(df, "rb") as df:
    #     WHD = pickle.load(df)

    homograph_lex = homograph_lex

    with open("WHD_nnvb_analysis/wordids.txt", "r") as wids:
        with open("WHD_nnvb_analysis/homographs.txt", "r") as homos:
            word_ids = wids.readlines()
            homographs = homos.readlines()

    word_ids = list(map(str.strip, word_ids))
    # get corresponding word_ids and homograph lists
    set_word_ids = list(set(map(str.strip, word_ids)))

    corresponding_homographs = []
    for id in set_word_ids:
        first_appearance_idx = find_indices(word_ids, id)[0]
        corresponding_homographs.append(homographs[first_appearance_idx])

    corresponding_homographs = list(map(str.strip, corresponding_homographs))

    assert len(set_word_ids) == len(corresponding_homographs)

    gt_pronunciations = []
    i = 0
    for wordid, homograph in zip(set_word_ids, corresponding_homographs):
        i += 1
        potential_prons = homograph_lex[homograph]
        print(i, "\t", wordid, "\t", homograph, "\t", potential_prons)
    #     chosen_pron_idx = int(input("Chosen pron idx: "))
    #     if chosen_pron_idx == "n":
    #         chosen_pron = input("Pron: ")
    #         gt_pronunciations.append(chosen_pron)
    #     else:
    #         gt_pronunciations.append(potential_prons[chosen_pron_idx])
    #     print(gt_pronunciations)
    #
    # print(gt_pronunciations)
    # input("OK?")
    # assert len(gt_pronunciations) == len(set_word_ids)
    #
    # with open("WHD_nnvb_analysis/gt_homograph_prons.txt", "w") as f:
    #     for pron in gt_pronunciations:
    #         f.write(pron)
    #         f.write("\n")

    # zipped_id_pron = list(zip(set_word_ids, gt_pronunciations))
    # df = pd.DataFrame(zipped_id_pron, columns=['wordid', 'gt_homograph_pron'])
    #
    # WHD = WHD.merge(df, on='wordid', how='left')
    # print(WHD.info())
    # print(WHD.head().to_string())
    # input("OK?")
    #
    # with open(f"WHD_FULL.pkl", "wb") as out_df:
    #     pickle.dump(WHD, out_df)


# get_gt_homograph_prons(get_lexicon("unilex-rpx.out"))


ood_sentence_idxs = [0, 3, 4, 6, 10, 15, 18, 19, 22, 24, 28, 32, 38, 39, 41, 42, 44, 45, 47, 48, 53, 57, 63, 69, 71, 77, 79, 82, 83, 86, 87, 88, 91, 93, 94, 95, 96, 99, 107, 109, 110, 115, 116, 117, 119, 120, 121, 123, 128, 130, 137, 142, 143, 144, 148, 149, 151, 152, 154, 156, 157, 159, 160, 163, 164, 166, 167, 168, 173, 182, 184, 193, 196, 200, 202, 204, 206, 207, 209, 214, 219, 221, 223, 225, 226, 229, 235, 241, 242, 245, 250, 251, 252, 255, 262, 263, 264, 265, 266, 268, 269, 273, 280, 285, 287, 290, 292, 295, 299, 301, 309, 317, 318, 321, 323, 329, 331, 333, 336, 342, 343, 347, 351, 356, 358, 361, 363, 367, 369, 371, 375, 385, 386, 388, 390, 391, 392, 396, 400, 406, 412, 414, 415, 416, 419, 420, 421, 422, 424, 425, 428, 429, 430, 434, 436, 442, 443, 446, 447, 453, 457, 458, 461, 465, 470, 472, 474, 475, 477, 480, 482, 483, 484, 485, 488, 489, 490, 491, 493, 495, 497, 499, 500, 501, 503, 506, 507, 510, 511, 512, 514, 515, 517, 519, 525, 529, 533, 536, 537, 538, 539, 540, 545, 546, 549, 554, 558, 562, 563, 571, 574, 576, 577, 585, 594, 595, 598, 599, 603, 610, 615, 623, 629, 630, 632, 633, 636, 638, 639, 641, 645, 653, 664, 668, 669, 673, 678, 681, 683, 688, 689, 692, 697, 700, 701, 703, 707, 708, 710, 711, 715, 716, 717, 718, 719, 721, 722, 732, 734, 738, 739, 740, 742, 744, 745, 746, 747, 752, 753, 755, 762, 763, 766, 774, 779, 781, 782, 783, 784, 785, 788, 790, 793, 794, 795, 807, 811, 812, 814, 816, 819, 823, 826, 827, 833, 838, 842, 844, 850, 852, 857, 858, 863, 864, 865, 869, 871, 872, 874, 876, 877, 878, 883, 885, 891, 893, 895, 901, 902, 903, 905, 909, 910, 915, 917, 922, 923, 924, 929, 935, 939, 940, 941, 942, 943, 944, 949, 952, 953, 955, 959, 964, 966, 968, 969, 977, 979, 980, 983, 984, 986, 988, 1001, 1002, 1003, 1004, 1007, 1008, 1012, 1013, 1015, 1018, 1020, 1021, 1024, 1028, 1029, 1030, 1031, 1032, 1035, 1040, 1042, 1044, 1046, 1048, 1050, 1051, 1057, 1058, 1069, 1076, 1080, 1082, 1084, 1094, 1098, 1103, 1105, 1107, 1108, 1110, 1111, 1114, 1115, 1119, 1120, 1121, 1124, 1127, 1129, 1134, 1143, 1152, 1155, 1156, 1157, 1159, 1160, 1161, 1168, 1171, 1177, 1179, 1180, 1181, 1184, 1185, 1187, 1188, 1189, 1192, 1194, 1198, 1199, 1200, 1204, 1208, 1211, 1212, 1213, 1214, 1216, 1217, 1223, 1226, 1229, 1234, 1235, 1238, 1242, 1248, 1249, 1254, 1255, 1256, 1257, 1262, 1263, 1266, 1268, 1271, 1272, 1273, 1275, 1279, 1288, 1289, 1297, 1300, 1301, 1305, 1308, 1312, 1313, 1315, 1317, 1325, 1326, 1327, 1332, 1333, 1334, 1339, 1341, 1342, 1344, 1346, 1349, 1353, 1355, 1357, 1360, 1361, 1363, 1367, 1368, 1371, 1373, 1375, 1377, 1378, 1382, 1386, 1387, 1389, 1390, 1391, 1395, 1397, 1400, 1401, 1402, 1404, 1405, 1407, 1408, 1410, 1411, 1412, 1415, 1416, 1417, 1418, 1420, 1424, 1425, 1426, 1429, 1436, 1439, 1440, 1441, 1449, 1452, 1453, 1455, 1457, 1461, 1468, 1471, 1474, 1476, 1481, 1486, 1492, 1493, 1494, 1497, 1499, 1501, 1503, 1505, 1508, 1513, 1514, 1517, 1522, 1527, 1530, 1543, 1549, 1550, 1552, 1556, 1560, 1565, 1576, 1583, 1586, 1587, 1589, 1591, 1593, 1594, 1596, 1598, 1603, 1610, 1611, 1619, 1621, 1622, 1623, 1626, 1631, 1632, 1634, 1643, 1645, 1646, 1655, 1661, 1671, 1674, 1683, 1686, 1688, 1690, 1699, 1704, 1708, 1710, 1711, 1713, 1714, 1721, 1726, 1730, 1734, 1736, 1739, 1744, 1745, 1752, 1759, 1768, 1769, 1780, 1783, 1789, 1792, 1793, 1795, 1797, 1801, 1804, 1805, 1808, 1810, 1811, 1814, 1815, 1819, 1820, 1822, 1823, 1824, 1825, 1826, 1829, 1831, 1833, 1834, 1835, 1843, 1844, 1846, 1851, 1852, 1853, 1856, 1860, 1861, 1862, 1863, 1865, 1867, 1870, 1872, 1874, 1877, 1880, 1881, 1883, 1885, 1889, 1897, 1898, 1902, 1904, 1906, 1907, 1911, 1913, 1920, 1926, 1928, 1930, 1937, 1939, 1942, 1943, 1949, 1952, 1954, 1955, 1956, 1957, 1958, 1962, 1964, 1971, 1972, 1974, 1980, 1984, 1995, 2003, 2004, 2008, 2014, 2018, 2021, 2023, 2026, 2029, 2033, 2034, 2035, 2039, 2042, 2043, 2046, 2047, 2052, 2059, 2061, 2062, 2067, 2069, 2075, 2077, 2078, 2079, 2083, 2086, 2087, 2088, 2090, 2093, 2094, 2095, 2097, 2098, 2099, 2100, 2104, 2106, 2108, 2109, 2117, 2118, 2119, 2120, 2121, 2123, 2124, 2126, 2137, 2140, 2141, 2149, 2152, 2153, 2154, 2155, 2157, 2158, 2171, 2176, 2185, 2188, 2192, 2194, 2195, 2196, 2202, 2203, 2204, 2205, 2210, 2214, 2216, 2226, 2228, 2231, 2235, 2243, 2250, 2251, 2256, 2257, 2261, 2264, 2266, 2267, 2268, 2269, 2275, 2278, 2279, 2283, 2285, 2288, 2290, 2291, 2303, 2306, 2309, 2310, 2311, 2313, 2315, 2318, 2320, 2323, 2330, 2331, 2334, 2339, 2343, 2345, 2348, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2360, 2363, 2364, 2368, 2370, 2371, 2372, 2373, 2374, 2376, 2377, 2378, 2380, 2381, 2382, 2385, 2388, 2389, 2392, 2393, 2397, 2398, 2400, 2405, 2416, 2422, 2423, 2424, 2428, 2433, 2434, 2435, 2438, 2439, 2441, 2442, 2443, 2445, 2449, 2450, 2451, 2453, 2454, 2459, 2460, 2464, 2465, 2466, 2468, 2470, 2472, 2473, 2474, 2479, 2480, 2481, 2482, 2483, 2485, 2486, 2487, 2488, 2493, 2495, 2497, 2499, 2500, 2507, 2508, 2510, 2515, 2519, 2520, 2523, 2524, 2529, 2534, 2535, 2536, 2539, 2540, 2543, 2553, 2554, 2556, 2559, 2565, 2566, 2567, 2570, 2572, 2574, 2575, 2580, 2581, 2590, 2591, 2592, 2593, 2594, 2595, 2598, 2602, 2604, 2605, 2606, 2607, 2615, 2616, 2623, 2624, 2626, 2630, 2636, 2641, 2642, 2644, 2646, 2648, 2650, 2655, 2657, 2658, 2659, 2662, 2663, 2664, 2667, 2668, 2669, 2673, 2674, 2683, 2684, 2686, 2688, 2690, 2699, 2701, 2703, 2706, 2707, 2709, 2712, 2713, 2714, 2715, 2720, 2728, 2730, 2737, 2738, 2744, 2753, 2755, 2761, 2762, 2763, 2766, 2768, 2770, 2772, 2773, 2775, 2777, 2780, 2789, 2790, 2792, 2793, 2796, 2797, 2801, 2805, 2807, 2808, 2810, 2815, 2816, 2821, 2824, 2826, 2835, 2836, 2840, 2841, 2844, 2850, 2853, 2857, 2860, 2861, 2865, 2871, 2872, 2876, 2878, 2880, 2881, 2882, 2885, 2907, 2908, 2913, 2915, 2917, 2918, 2919, 2920, 2921, 2940, 2942, 2946, 2949, 2952, 2953, 2955, 2956, 2957, 2959, 2961, 2964, 2965, 2969, 2971, 2972, 2976, 2983, 2985, 2987, 2990, 2991, 2993, 2996, 2998, 2999, 3003, 3006, 3009, 3010, 3012, 3015, 3017, 3022, 3025, 3026, 3028, 3032, 3033, 3035, 3036, 3037, 3044, 3047, 3051, 3055, 3057, 3058, 3059, 3061, 3062, 3065, 3067, 3068, 3070, 3074, 3076, 3079, 3083, 3084, 3086, 3087, 3095, 3096, 3099, 3100, 3101, 3102, 3104, 3109, 3112, 3113, 3116, 3125, 3128, 3137, 3140, 3141, 3143, 3145, 3148, 3149, 3154, 3156, 3157, 3159, 3160, 3162, 3163, 3164, 3171, 3172, 3174, 3175, 3177, 3180, 3182, 3183, 3185, 3186, 3187, 3188, 3192, 3194, 3195, 3197, 3199, 3200, 3202, 3203, 3204, 3208, 3210, 3213, 3214, 3218, 3222, 3223, 3226, 3227, 3228, 3231, 3234, 3235, 3240, 3241, 3249, 3252, 3254, 3255, 3260, 3261, 3265, 3266, 3267, 3268, 3269, 3272, 3276, 3278, 3281, 3283, 3285, 3286, 3288, 3289, 3294, 3308, 3309, 3311, 3312, 3313, 3315, 3316, 3320, 3322, 3326, 3333, 3338, 3343, 3349, 3350, 3352, 3356, 3357, 3360, 3363, 3364, 3372, 3374, 3378, 3386, 3387, 3390, 3391, 3396, 3402, 3404, 3407, 3410, 3411, 3414, 3415, 3420, 3421, 3429, 3431, 3432, 3435, 3442, 3443, 3444, 3445, 3446, 3448, 3449, 3454, 3460, 3461, 3463, 3465, 3468, 3473, 3474, 3475, 3481, 3482, 3483, 3484, 3487, 3488, 3489, 3491, 3492, 3494, 3500, 3502, 3510, 3512, 3516, 3520, 3521, 3527, 3528, 3530, 3532, 3534, 3536, 3537, 3539, 3540, 3542, 3543, 3545, 3546, 3547, 3548, 3549, 3553, 3559, 3561, 3563, 3564, 3567, 3568, 3569, 3571, 3572, 3576, 3578, 3580, 3581, 3582, 3585, 3586, 3588, 3589, 3591, 3592, 3594, 3598, 3600, 3602, 3611, 3612, 3614, 3615, 3616, 3617, 3623, 3624, 3625, 3627, 3636, 3637, 3642, 3645, 3647, 3650, 3652, 3653, 3654, 3657, 3658, 3662, 3664, 3665, 3675, 3683, 3688, 3695, 3697, 3704, 3709, 3713, 3714, 3720, 3725, 3731, 3733, 3735, 3738, 3743, 3749, 3751, 3754, 3760, 3763, 3769, 3781, 3791, 3799, 3808, 3811, 3812, 3819, 3821, 3823, 3824, 3837, 3839, 3850, 3851, 3857, 3863, 3864, 3865, 3867, 3868, 3869, 3870, 3871, 3874, 3877, 3879, 3880, 3883, 3889, 3890, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3904, 3907, 3909, 3911, 3913, 3921, 3926, 3927, 3930, 3932, 3933, 3935, 3938, 3945, 3946, 3947, 3949, 3953, 3954, 3957, 3961, 3967, 3969, 3970, 3972, 3973, 3978, 3982, 3984, 3985, 3989, 3991, 3992, 4000, 4002, 4003, 4004, 4005, 4008, 4010, 4018, 4022, 4027, 4028, 4029, 4032, 4033, 4034, 4036, 4037, 4040, 4042, 4047, 4048, 4050, 4055, 4066, 4076, 4079, 4080, 4081, 4082, 4083, 4088, 4089, 4091, 4093, 4099, 4100, 4102, 4105, 4111, 4113, 4115, 4117, 4119, 4120, 4123, 4126, 4129, 4131, 4133, 4135, 4136, 4137, 4139, 4142, 4144, 4145, 4147, 4148, 4149, 4150, 4154, 4160, 4162, 4163, 4166, 4168, 4170, 4171, 4173, 4174, 4182, 4183, 4184, 4185, 4187, 4189, 4194, 4199, 4200, 4203, 4204, 4210, 4214, 4215, 4218, 4221, 4225, 4229, 4230, 4232, 4237, 4241, 4247, 4249, 4254, 4256, 4257, 4258, 4259, 4264, 4266, 4268, 4269, 4274, 4277, 4280, 4282, 4291, 4293, 4301, 4303, 4304, 4308, 4309, 4310, 4311, 4321, 4329, 4334, 4336, 4337, 4343, 4355, 4363, 4365, 4366, 4368, 4370, 4373, 4374, 4381, 4382, 4383, 4384, 4385, 4386, 4387, 4390, 4391, 4400, 4404, 4406, 4408, 4413, 4414, 4415, 4416, 4419, 4420, 4422, 4427, 4430, 4433, 4438, 4439, 4441, 4442, 4444, 4448, 4449, 4454, 4456, 4461, 4470, 4472, 4474, 4475, 4476, 4478, 4479, 4480, 4485, 4487, 4490, 4494, 4495, 4501, 4502, 4506, 4508, 4509, 4510, 4513, 4515, 4516, 4519, 4520, 4522, 4526, 4528, 4529, 4530, 4533, 4536, 4543, 4544, 4546, 4553, 4554, 4555, 4557, 4558, 4559, 4561, 4564, 4565, 4568, 4573, 4574, 4581, 4582, 4587, 4590, 4591, 4596, 4598, 4599, 4601, 4603, 4604, 4605, 4606, 4607, 4609, 4611, 4613, 4614, 4616, 4617, 4619, 4621, 4622, 4626, 4629, 4635, 4638, 4641, 4642, 4643, 4645, 4647, 4648, 4651, 4655, 4657, 4662, 4664, 4667, 4671, 4672, 4677, 4681, 4685, 4686, 4691, 4693, 4696, 4699, 4703, 4707, 4709, 4713, 4715, 4716, 4719, 4721, 4722, 4724, 4727, 4735, 4741, 4744, 4752, 4760, 4762, 4764, 4765, 4768, 4770, 4771, 4775, 4776, 4781, 4783, 4787, 4799, 4806, 4808, 4809, 4813, 4814, 4825, 4826, 4828, 4831, 4835, 4836, 4838, 4851, 4853, 4856, 4857, 4858, 4860, 4861, 4863, 4867, 4868, 4869, 4873, 4874, 4875, 4880, 4883, 4884, 4890, 4901, 4906, 4908, 4910, 4917, 4921, 4927, 4928, 4929, 4931, 4933, 4935, 4936, 4937, 4940, 4942, 4945, 4949, 4953, 4956, 4963, 4964, 4968, 4969, 4971, 4973, 4982, 4984, 4986, 4987, 4988, 4990, 4997, 5003, 5010, 5014, 5017, 5018, 5021, 5022, 5025, 5027, 5036, 5037, 5039, 5044, 5050, 5061, 5063, 5064, 5066, 5069, 5072, 5074, 5075, 5076, 5081, 5083, 5092, 5095, 5097, 5100, 5104, 5106, 5107, 5111, 5115, 5116, 5120, 5123, 5124, 5125, 5130, 5134, 5135, 5136, 5139, 5142, 5144, 5146, 5147, 5150, 5151, 5153, 5165, 5166, 5175, 5176, 5178, 5182, 5184, 5185, 5186, 5189, 5191, 5192, 5194, 5195, 5199, 5200, 5203, 5211, 5215, 5217, 5218, 5219, 5221, 5225, 5226, 5229, 5230, 5231, 5234, 5238, 5239, 5243, 5244, 5245, 5251, 5255, 5256, 5257, 5265, 5269, 5273, 5274, 5275, 5276, 5278, 5281, 5286, 5287, 5288, 5290, 5297, 5299, 5302, 5303, 5304, 5306, 5308, 5309, 5321, 5324, 5325, 5328, 5331, 5332, 5337, 5341, 5343, 5344, 5345, 5348, 5352, 5357, 5359, 5364, 5365, 5366, 5367, 5372, 5378, 5383, 5394, 5397, 5399, 5408, 5409, 5411, 5413, 5416, 5420, 5421, 5424, 5426, 5429, 5430, 5434, 5437, 5438, 5440, 5441, 5443, 5444, 5449, 5451, 5456, 5463, 5467, 5474, 5480, 5484, 5488, 5499, 5505, 5507, 5508, 5509, 5514, 5516, 5517, 5518, 5519, 5521, 5525, 5528, 5533, 5535, 5538, 5541, 5543, 5546, 5547, 5551, 5554, 5556, 5557, 5558, 5562, 5563, 5564, 5571, 5572, 5574, 5578, 5579, 5581, 5586, 5588, 5590, 5595, 5602, 5606, 5607, 5609, 5610, 5611, 5613, 5617, 5618, 5620, 5623, 5625, 5626, 5630, 5634, 5636, 5646, 5648, 5652, 5656, 5661, 5665, 5676, 5680, 5687, 5692, 5693, 5702, 5707, 5713, 5714, 5721, 5722, 5728, 5737, 5740, 5746, 5752, 5758, 5760, 5762, 5764, 5766, 5767, 5773, 5774, 5778, 5780, 5783, 5784, 5785, 5786, 5794, 5801, 5806, 5813, 5819, 5822, 5824, 5833, 5837, 5843, 5846, 5847, 5851, 5852, 5861, 5863, 5871, 5872, 5881, 5883, 5884, 5886, 5888, 5890, 5891, 5899, 5900, 5906, 5908, 5910, 5911, 5912, 5913, 5918, 5919, 5925, 5930, 5932, 5934, 5942, 5948, 5949, 5952, 5954, 5955, 5957, 5961, 5974, 5975, 5977, 5980, 5983, 5986, 5988, 5990, 5992, 5998, 6004, 6009, 6010, 6014, 6016, 6019, 6023, 6024, 6028, 6031, 6033, 6036, 6038, 6040, 6041, 6047, 6049, 6050, 6052, 6053, 6055, 6057, 6065, 6068, 6076, 6078, 6081, 6082, 6084, 6088, 6093, 6095, 6098, 6100, 6104, 6106, 6107, 6112, 6114, 6115, 6116, 6117, 6118, 6121, 6125, 6127, 6129, 6130, 6131, 6135, 6136, 6138, 6143, 6145, 6147, 6149, 6151, 6152, 6154, 6156, 6157, 6161, 6163, 6164, 6166, 6167, 6169, 6172, 6174, 6179, 6180, 6182, 6183, 6184, 6185, 6186, 6187, 6190, 6191, 6192, 6196, 6197, 6198, 6199, 6200, 6201, 6202, 6203, 6208, 6210, 6212, 6214, 6220, 6223, 6226, 6227, 6228, 6234, 6235, 6238, 6244, 6245, 6247, 6252, 6253, 6255, 6257, 6259, 6261, 6266, 6268, 6269, 6270, 6271, 6274, 6275, 6276, 6277, 6279, 6284, 6287, 6288, 6289, 6291, 6292, 6294, 6295, 6296, 6299, 6300, 6301, 6302, 6303, 6304, 6305, 6306, 6308, 6309, 6310, 6314, 6318, 6319, 6320, 6321, 6322, 6324, 6325, 6327, 6329, 6330, 6331, 6332, 6334, 6335, 6336, 6337, 6338, 6339, 6340, 6341, 6343, 6344, 6345, 6346, 6347, 6348, 6349, 6350, 6353, 6361, 6362, 6367, 6376, 6377, 6379, 6380, 6382, 6385, 6389, 6393, 6395, 6402, 6403, 6406, 6408, 6411, 6417, 6419, 6423, 6424, 6427, 6438, 6443, 6444, 6447, 6448, 6450, 6452, 6454, 6457, 6458, 6461, 6463, 6467, 6468, 6469, 6476, 6477, 6481, 6482, 6483, 6487, 6497, 6498, 6500, 6503, 6504, 6511, 6512, 6516, 6518, 6521, 6527, 6529, 6531, 6533, 6534, 6540, 6544, 6547, 6549, 6551, 6552, 6553, 6554, 6555, 6557, 6559, 6562, 6563, 6564, 6566, 6570, 6571, 6572, 6574, 6575, 6576, 6577, 6578, 6582, 6590, 6597, 6600, 6607, 6609, 6611, 6612, 6614, 6618, 6619, 6621, 6625, 6627, 6630, 6632, 6639, 6640, 6643, 6644, 6648, 6651, 6654, 6655, 6658, 6659, 6663, 6665, 6671, 6672, 6673, 6674, 6675, 6677, 6683, 6684, 6686, 6687, 6691, 6695, 6697, 6699, 6713, 6715, 6716, 6719, 6720, 6723, 6724, 6725, 6726, 6727, 6728, 6729, 6731, 6734, 6735, 6737, 6738, 6740, 6743, 6745, 6747, 6753, 6756, 6760, 6764, 6765, 6773, 6777, 6779, 6785, 6788, 6792, 6799, 6801, 6803, 6806, 6808, 6809, 6813, 6814, 6816, 6817, 6821, 6823, 6824, 6826, 6827, 6831, 6837, 6839, 6840, 6851, 6853, 6862, 6867, 6872, 6874, 6875, 6880, 6887, 6890, 6892, 6895, 6896, 6899, 6900, 6901, 6902, 6903, 6908, 6913, 6914, 6915, 6916, 6918, 6920, 6922, 6923, 6930, 6932, 6933, 6935, 6939, 6942, 6943, 6945, 6947, 6948, 6950, 6952, 6957, 6959, 6963, 6965, 6969, 6971, 6975, 6976, 6977, 6978, 6979, 6980, 6982, 6984, 6985, 6990, 6997, 7004, 7014, 7016, 7017, 7018, 7025, 7031, 7034, 7037, 7038, 7039, 7042, 7043, 7045, 7046, 7047, 7050, 7053, 7055, 7077, 7080, 7082, 7085, 7087, 7089, 7100, 7104, 7108, 7109, 7110, 7114, 7116, 7120, 7134, 7139, 7143, 7145, 7148, 7150, 7157, 7161, 7162, 7164, 7171, 7172, 7181, 7182, 7183, 7184, 7187, 7192, 7193, 7194, 7199, 7203, 7212, 7215, 7216, 7222, 7227, 7230, 7232, 7235, 7236, 7237, 7239, 7241, 7242, 7243, 7245, 7253, 7254, 7255, 7261, 7262, 7263, 7264, 7265, 7267, 7273, 7274, 7276, 7277, 7279, 7281, 7283, 7284, 7286, 7287, 7290, 7292, 7296, 7299, 7301, 7302, 7307, 7310, 7311, 7315, 7317, 7318, 7320, 7321, 7324, 7325, 7328, 7329, 7330, 7331, 7332, 7333, 7335, 7338, 7339, 7340, 7346, 7348, 7352, 7353, 7354, 7357, 7358, 7360, 7364, 7367, 7369, 7371, 7372, 7374, 7377, 7378, 7380, 7382, 7383, 7386, 7387, 7390, 7391, 7392, 7403, 7406, 7414, 7420, 7426, 7428, 7432, 7433, 7434, 7436, 7439, 7442, 7443, 7448, 7449, 7450, 7451, 7454, 7457, 7463, 7464, 7467, 7468, 7469, 7470, 7471, 7472, 7473, 7474, 7475, 7476, 7477, 7480, 7481, 7482, 7489, 7493, 7497, 7503, 7506, 7507, 7508, 7510, 7511, 7512, 7514, 7517, 7518, 7519, 7521, 7525, 7535, 7536, 7539, 7543, 7545, 7546, 7548, 7549, 7556, 7564, 7566, 7569, 7578, 7579, 7580, 7581, 7590, 7599, 7602, 7609, 7611, 7613, 7615, 7618, 7621, 7622, 7623, 7624, 7625, 7627, 7632, 7634, 7635, 7636, 7637, 7638, 7641, 7642, 7643, 7644, 7645, 7646, 7648, 7654, 7655, 7657, 7661, 7662, 7664, 7665, 7666, 7668, 7670, 7671, 7674, 7675, 7676, 7677, 7678, 7679, 7680, 7681, 7682, 7683, 7687, 7688, 7689, 7690, 7692, 7694, 7695, 7696, 7699, 7700, 7702, 7709, 7712, 7713, 7715, 7718, 7725, 7733, 7734, 7736, 7737, 7740, 7742, 7746, 7750, 7751, 7756, 7758, 7762, 7767, 7770, 7775, 7776, 7795, 7800, 7804, 7817, 7822, 7834, 7850, 7858, 7865, 7870, 7872, 7874, 7875, 7876, 7880, 7883, 7889, 7892, 7893, 7894, 7895, 7896, 7902, 7906, 7917, 7919, 7920, 7923, 7929, 7935, 7940, 7947, 7949, 7958, 7959, 7960, 7961, 7968, 7972, 7974, 7978, 7982, 7983, 7986, 7987, 7989, 7990, 7991, 7992, 7994, 7996, 7999, 8000, 8002, 8005, 8007, 8008, 8009, 8010, 8016, 8017, 8020, 8021, 8023, 8024, 8025, 8026, 8027, 8031, 8032, 8035, 8036, 8037, 8043, 8044, 8047, 8050, 8051, 8059, 8064, 8068, 8072, 8079, 8081, 8084, 8087, 8096, 8098, 8100, 8103, 8106, 8107, 8116, 8123, 8127, 8132, 8133, 8135, 8143, 8145, 8146, 8147, 8148, 8150, 8153, 8155, 8156, 8157, 8158, 8160, 8162, 8164, 8165, 8166, 8167, 8169, 8170, 8172, 8174, 8178, 8179, 8180, 8181, 8183, 8186, 8187, 8189, 8190, 8191, 8195, 8198, 8200, 8203, 8204, 8205, 8206, 8207, 8210, 8212, 8213, 8215, 8216, 8217, 8218, 8219, 8222, 8223, 8226, 8227, 8231, 8236, 8238, 8241, 8246, 8248, 8249, 8258, 8269, 8272, 8275, 8283, 8284, 8287, 8289, 8292, 8298, 8300, 8306, 8311, 8322, 8327, 8328, 8330, 8333, 8336, 8339, 8341, 8342, 8348, 8350, 8354, 8359, 8361, 8367, 8368, 8369, 8375, 8377, 8379, 8382, 8383, 8388, 8390, 8392, 8394, 8396, 8398, 8404, 8405, 8408, 8412, 8416, 8418, 8420, 8421, 8422, 8426, 8437, 8438, 8440, 8441, 8449, 8455, 8457, 8459, 8460, 8462, 8465, 8466, 8467, 8468, 8469, 8471, 8472, 8475, 8477, 8483, 8486, 8487, 8498, 8503, 8505, 8511, 8512, 8513, 8514, 8515, 8517, 8520, 8521, 8523, 8524, 8525, 8529, 8531, 8532, 8534, 8536, 8537, 8538, 8539, 8541, 8542, 8543, 8544, 8548, 8549, 8551, 8552, 8554, 8555, 8556, 8557, 8558, 8559, 8562, 8566, 8570, 8571, 8573, 8574, 8582, 8585, 8586, 8588, 8589, 8594, 8595, 8600, 8604, 8605, 8606, 8607, 8610, 8613, 8614, 8620, 8621, 8623, 8625, 8627, 8628, 8629, 8631, 8632, 8633, 8634, 8635, 8638, 8639, 8641, 8646, 8654, 8661, 8664, 8666, 8671, 8673, 8674, 8675, 8677, 8680, 8683, 8688, 8689, 8690, 8691, 8704, 8705, 8706, 8707, 8709, 8714, 8715, 8718, 8719, 8720, 8723, 8726, 8729, 8734, 8735, 8737, 8739, 8740, 8741, 8744, 8752, 8761, 8762, 8764, 8765, 8768, 8771, 8772, 8779, 8780, 8784, 8790, 8798, 8803, 8806, 8808, 8809, 8810, 8813, 8815, 8816, 8819, 8821, 8822, 8824, 8826, 8828, 8838, 8839, 8845, 8846, 8851, 8852, 8855, 8856, 8859, 8861, 8864, 8865, 8867, 8873, 8874, 8884, 8890, 8892, 8897, 8899, 8909, 8910, 8915, 8916, 8919, 8920, 8922, 8925, 8926, 8932, 8934, 8937, 8938, 8939, 8941, 8942, 8944, 8949, 8952, 8953, 8954, 8955, 8957, 8958, 8959, 8960, 8963, 8967, 8968, 8969, 8972, 8974, 8976, 8977, 8979, 8982, 8984, 8986, 8987, 8994, 8996, 8997, 8998, 9000, 9001, 9002, 9003, 9004, 9007, 9011, 9013, 9017, 9020, 9026, 9027, 9029, 9034, 9035, 9036, 9037, 9038, 9039, 9042, 9043, 9047, 9050, 9054, 9063, 9076, 9077, 9080, 9083, 9085, 9087, 9095, 9100, 9101, 9102, 9103, 9104, 9105, 9107, 9114, 9115, 9118, 9119, 9121, 9122, 9125, 9127, 9128, 9129, 9130, 9132, 9135, 9138, 9140, 9141, 9142, 9143, 9144, 9145, 9146, 9151, 9154, 9155, 9156, 9157, 9158, 9159, 9161, 9164, 9165, 9168, 9172, 9173, 9175, 9181, 9183, 9184, 9185, 9188, 9191, 9194, 9195, 9200, 9201, 9210, 9213, 9219, 9224, 9225, 9226, 9228, 9229, 9230, 9240, 9242, 9246, 9248, 9249, 9251, 9256, 9260, 9264, 9266, 9268, 9270, 9271, 9277, 9278, 9280, 9286, 9288, 9289, 9290, 9291, 9299, 9311, 9318, 9321, 9323, 9324, 9327, 9330, 9337, 9345, 9346, 9348, 9353, 9355, 9367, 9371, 9373, 9375, 9376, 9378, 9379, 9386, 9387, 9393, 9396, 9397, 9400, 9403, 9407, 9408, 9411, 9418, 9420, 9422, 9423, 9433, 9435, 9436, 9443, 9451, 9456, 9457, 9460, 9461, 9463, 9472, 9479, 9480, 9481, 9484, 9490, 9498, 9499, 9504, 9505, 9508, 9509, 9513, 9516, 9517, 9518, 9521, 9526, 9527, 9529, 9534, 9535, 9536, 9539, 9542, 9546, 9548, 9549, 9556, 9557, 9561, 9562, 9563, 9565, 9566, 9568, 9569, 9570, 9573, 9574, 9575, 9576, 9579, 9581, 9582, 9585, 9588, 9594, 9596, 9598, 9604, 9609, 9610, 9611, 9612, 9613, 9618, 9619, 9620, 9621, 9624, 9625, 9627, 9630, 9639, 9641, 9642, 9646, 9647, 9649, 9652, 9655, 9657, 9660, 9662, 9665, 9673, 9674, 9678, 9679, 9682, 9684, 9693, 9694, 9696, 9700, 9701, 9707, 9714, 9718, 9720, 9722, 9723, 9729, 9730, 9733, 9734, 9738, 9739, 9740, 9742, 9744, 9745, 9747, 9753, 9754, 9756, 9758, 9764, 9765, 9768, 9770, 9774, 9778, 9779, 9780, 9784, 9785, 9788, 9791, 9794, 9795, 9797, 9798, 9800, 9805, 9807, 9813, 9817, 9819, 9820, 9821, 9822, 9825, 9828, 9829, 9831, 9833, 9834, 9835, 9837, 9839, 9840, 9842, 9843, 9844, 9847, 9850, 9855, 9856, 9860, 9861, 9862, 9865, 9866, 9867, 9868, 9869, 9870, 9874, 9877, 9878, 9879, 9881, 9882, 9883, 9884, 9885, 9886, 9889, 9896, 9898, 9899, 9900, 9916, 9917, 9921, 9922, 9928, 9935, 9937, 9944, 9946, 9948, 9951, 9952, 9957, 9959, 9963, 9972, 9974, 9979, 9984, 9985, 9987, 9989, 9993, 9994, 9995, 9999, 10004, 10005, 10012, 10014, 10017, 10020, 10021, 10025, 10029, 10030, 10038, 10039, 10042, 10044, 10045, 10048, 10061, 10066, 10068, 10073, 10077, 10078, 10085, 10094, 10096, 10097, 10099, 10101, 10102, 10108, 10111, 10114, 10117, 10119, 10121, 10123, 10124, 10131, 10136, 10141, 10143, 10149, 10152, 10154, 10155, 10156, 10159, 10164, 10166, 10167, 10178, 10181, 10189, 10190, 10199, 10200, 10205, 10206, 10207, 10208, 10218, 10221, 10223, 10224, 10230, 10231, 10237, 10238, 10242, 10244, 10245, 10246, 10248, 10251, 10253, 10258, 10259, 10260, 10261, 10264, 10266, 10270, 10271, 10272, 10274, 10276, 10278, 10279, 10280, 10282, 10284, 10286, 10295, 10299, 10301, 10303, 10304, 10305, 10306, 10308, 10312, 10313, 10314, 10317, 10319, 10323, 10324, 10329, 10331, 10334, 10335, 10338, 10341, 10342, 10346, 10347, 10351, 10354, 10355, 10357, 10358, 10360, 10361, 10362, 10374, 10375, 10378, 10379, 10386, 10387, 10390, 10391, 10396, 10402, 10403, 10405, 10408, 10409, 10410, 10412, 10414, 10416, 10418, 10419, 10421, 10422, 10424, 10431, 10432, 10438, 10441, 10443, 10446, 10448, 10449, 10450, 10451, 10453, 10454, 10455, 10456, 10457, 10458, 10459, 10460, 10462, 10464, 10467, 10468, 10469, 10470, 10472, 10474, 10475, 10476, 10477, 10478, 10479, 10480, 10481, 10482, 10483, 10484, 10485, 10487, 10488, 10489, 10490, 10491, 10492, 10493, 10496, 10497, 10498, 10499, 10500, 10502, 10504, 10505, 10508, 10509, 10511, 10512, 10514, 10517, 10518, 10519, 10520, 10521, 10522, 10523, 10524, 10526, 10527, 10529, 10530, 10531, 10532, 10533, 10539, 10540, 10541, 10543, 10544, 10545, 10547, 10548, 10551, 10553, 10554, 10555, 10556, 10558, 10565, 10570, 10572, 10576, 10578, 10580, 10581, 10584, 10586, 10587, 10591, 10595, 10596, 10597, 10599, 10601, 10603, 10605, 10609, 10610, 10614, 10616, 10617, 10619, 10621, 10622, 10623, 10624, 10628, 10643, 10644, 10650, 10651, 10657, 10658, 10660, 10662, 10665, 10667, 10669, 10671, 10673, 10678, 10680, 10682, 10686, 10688, 10689, 10691, 10696, 10699, 10701, 10703, 10707, 10712, 10715, 10716, 10717, 10718, 10719, 10722, 10724, 10725, 10727, 10728, 10729, 10730, 10731, 10733, 10737, 10739, 10742, 10743, 10747, 10752, 10760, 10763, 10764, 10765, 10766, 10767, 10768, 10770, 10771, 10772, 10773, 10774, 10775, 10776, 10777, 10778, 10787, 10790, 10798, 10800, 10802, 10805, 10806, 10807, 10809, 10813, 10814, 10815, 10816, 10823, 10827, 10828, 10829, 10830, 10831, 10832, 10833, 10839, 10842, 10844, 10848, 10853, 10856, 10857, 10859, 10862, 10864, 10866, 10869, 10870, 10872, 10878, 10880, 10882, 10883, 10884, 10886, 10889, 10891, 10897, 10898, 10901, 10902, 10904, 10906, 10907, 10910, 10913, 10914, 10917, 10919, 10921, 10924, 10925, 10926, 10928, 10929, 10933, 10936, 10937, 10938, 10940, 10941, 10959, 10960, 10961, 10962, 10963, 10964, 10969, 10971, 10973, 10975, 10977, 10980, 10983, 10992, 10997, 11003, 11010, 11011, 11013, 11014, 11019, 11020, 11026, 11033, 11036, 11040, 11044, 11047, 11048, 11053, 11054, 11056, 11057, 11058, 11061, 11065, 11067, 11068, 11071, 11072, 11074, 11078, 11080, 11082, 11089, 11090, 11094, 11095, 11096, 11097, 11104, 11106, 11109, 11113, 11114, 11122, 11124, 11126, 11128, 11130, 11132, 11134, 11136, 11137, 11138, 11141, 11142, 11143, 11145, 11149, 11150, 11152, 11153, 11155, 11156, 11159, 11162, 11169, 11171, 11172, 11174, 11175, 11178, 11179, 11180, 11182, 11183, 11185, 11186, 11187, 11188, 11189, 11190, 11191, 11198, 11199, 11203, 11204, 11206, 11207, 11208, 11211, 11212, 11213, 11214, 11219, 11222, 11223, 11224, 11225, 11229, 11230, 11231, 11238, 11239, 11241, 11244, 11246, 11247, 11250, 11252, 11255, 11260, 11263, 11264, 11271, 11274, 11279, 11283, 11286, 11288, 11289, 11290, 11299, 11301, 11304, 11305, 11308, 11311, 11316, 11320, 11325, 11328, 11333, 11334, 11336, 11337, 11339, 11340, 11342, 11346, 11348, 11351, 11353, 11354, 11358, 11360, 11363, 11368, 11371, 11381, 11383, 11386, 11388, 11389, 11392, 11393, 11395, 11396, 11398, 11400, 11401, 11404, 11412, 11413, 11417, 11422, 11428, 11429, 11432, 11433, 11435, 11436, 11446, 11452, 11453, 11455, 11462, 11466, 11467, 11469, 11470, 11471, 11472, 11475, 11482, 11483, 11485, 11488, 11489, 11493, 11496, 11497, 11500, 11502, 11503, 11504, 11506, 11507, 11508, 11513, 11514, 11515, 11519, 11520, 11523, 11527, 11528, 11529, 11531, 11532, 11534, 11536, 11543, 11544, 11546, 11548, 11551, 11552, 11557, 11559, 11561, 11565, 11569, 11570, 11573, 11576, 11585, 11586, 11587, 11588, 11589, 11590, 11605, 11607, 11609, 11611, 11613, 11618, 11619, 11621, 11622, 11623, 11624, 11625, 11626, 11627, 11628, 11630, 11631, 11632, 11636, 11637, 11638, 11639, 11641, 11643, 11644, 11647, 11649, 11653, 11655, 11661, 11662, 11663, 11667, 11672, 11674, 11675, 11679, 11686, 11689, 11690, 11699, 11700, 11701, 11703, 11705, 11706, 11709, 11711, 11712, 11713, 11716, 11718, 11723, 11724, 11727, 11728, 11732, 11734, 11739, 11742, 11743, 11744, 11748, 11750, 11757, 11761, 11762, 11763, 11768, 11770, 11776, 11781, 11792, 11793, 11795, 11796, 11798, 11800, 11802, 11805, 11806, 11808, 11811, 11814, 11819, 11825, 11826, 11827, 11830, 11832, 11833, 11834, 11835, 11840, 11842, 11843, 11847, 11849, 11850, 11851, 11852, 11854, 11855, 11857, 11860, 11861, 11867, 11868, 11869, 11870, 11871, 11875, 11876, 11877, 11885, 11888, 11890, 11893, 11898, 11901, 11910, 11918, 11919, 11921, 11928, 11930, 11939, 11941, 11945, 11950, 11954, 11955, 11959, 11962, 11966, 11969, 11976, 11979, 11985, 11990, 11992, 11998, 11999, 12001, 12004, 12007, 12010, 12014, 12020, 12023, 12025, 12033, 12036, 12037, 12038, 12040, 12045, 12047, 12051, 12054, 12061, 12062, 12063, 12064, 12066, 12068, 12069, 12070, 12071, 12073, 12075, 12081, 12082, 12083, 12085, 12087, 12091, 12097, 12098, 12101, 12103, 12104, 12105, 12107, 12109, 12111, 12112, 12114, 12115, 12116, 12119, 12124, 12126, 12130, 12132, 12135, 12136, 12137, 12138, 12139, 12140, 12142, 12143, 12144, 12148, 12149, 12152, 12154, 12166, 12169, 12179, 12190, 12194, 12195, 12196, 12203, 12206, 12213, 12215, 12219, 12225, 12226, 12229, 12237, 12239, 12246, 12249, 12250, 12255, 12256, 12257, 12265, 12269, 12270, 12277, 12285, 12289, 12294, 12295, 12303, 12308, 12311, 12315, 12316, 12321, 12322, 12327, 12328, 12329, 12332, 12333, 12335, 12338, 12339, 12343, 12344, 12345, 12346, 12347, 12349, 12351, 12352, 12359, 12360, 12361, 12363, 12365, 12366, 12367, 12368, 12370, 12371, 12374, 12375, 12378, 12379, 12383, 12384, 12388, 12393, 12394, 12395, 12403, 12405, 12406, 12407, 12411, 12412, 12413, 12415, 12421, 12423, 12427, 12428, 12429, 12437, 12444, 12449, 12450, 12460, 12461, 12466, 12468, 12475, 12478, 12485, 12487, 12493, 12494, 12495, 12497, 12500, 12502, 12505, 12506, 12516, 12517, 12522, 12523, 12526, 12527, 12530, 12533, 12534, 12545, 12549, 12551, 12555, 12557, 12558, 12559, 12561, 12569, 12570, 12572, 12580, 12581, 12583, 12589, 12591, 12595, 12596, 12598, 12601, 12602, 12603, 12605, 12608, 12610, 12612, 12615, 12616, 12617, 12618, 12621, 12623, 12624, 12625, 12626, 12628, 12629, 12631, 12635, 12638, 12640, 12641, 12643, 12644, 12645, 12646, 12648, 12649, 12650, 12651, 12654, 12656, 12658, 12659, 12660, 12662, 12663, 12664, 12665, 12666, 12667, 12668, 12669, 12670, 12673, 12674, 12675, 12676, 12677, 12678, 12679, 12680, 12681, 12682, 12683, 12685, 12687, 12688, 12689, 12690, 12691, 12692, 12693, 12694, 12695, 12696, 12697, 12698, 12700, 12701, 12703, 12705, 12708, 12713, 12716, 12717, 12721, 12722, 12728, 12732, 12733, 12734, 12735, 12739, 12740, 12744, 12747, 12751, 12760, 12762, 12774, 12775, 12779, 12781, 12782, 12784, 12786, 12795, 12800, 12801, 12805, 12806, 12808, 12809, 12811, 12817, 12819, 12821, 12827, 12829, 12830, 12834, 12836, 12837, 12839, 12840, 12844, 12845, 12846, 12847, 12850, 12854, 12855, 12856, 12857, 12860, 12861, 12863, 12864, 12865, 12869, 12871, 12872, 12873, 12877, 12878, 12890, 12893, 12899, 12906, 12908, 12909, 12913, 12918, 12929, 12935, 12939, 12945, 12947, 12957, 12960, 12962, 12972, 12976, 12979, 12981, 12990, 12992, 12995, 12996, 12999, 13001, 13006, 13007, 13009, 13014, 13016, 13018, 13020, 13022, 13025, 13027, 13028, 13033, 13034, 13035, 13036, 13038, 13039, 13041, 13042, 13044, 13047, 13055, 13067, 13069, 13082, 13083, 13087, 13089, 13091, 13093, 13094, 13097, 13098, 13099, 13100, 13111, 13118, 13123, 13124, 13130, 13133, 13144, 13146, 13148, 13156, 13157, 13158, 13160, 13162, 13164, 13167, 13170, 13171, 13180, 13181, 13182, 13184, 13185, 13186, 13189, 13191, 13197, 13201, 13203, 13204, 13209, 13210, 13218, 13224, 13225, 13231, 13232, 13233, 13236, 13237, 13240, 13245, 13248, 13255, 13262, 13264, 13266, 13271, 13272, 13281, 13282, 13285, 13286, 13292, 13295, 13296, 13299, 13302, 13306, 13310, 13315, 13322, 13332, 13333, 13334, 13337, 13338, 13339, 13340, 13341, 13342, 13344, 13347, 13349, 13351, 13352, 13353, 13355, 13356, 13357, 13358, 13361, 13362, 13365, 13366, 13367, 13368, 13369, 13370, 13372, 13374, 13377, 13379, 13380, 13385, 13386, 13387, 13393, 13394, 13395, 13396, 13397, 13401, 13403, 13404, 13405, 13406, 13407, 13409, 13411, 13413, 13414, 13415, 13416, 13417, 13419, 13422, 13425, 13426, 13428, 13430, 13431, 13433, 13434, 13439, 13451, 13459, 13460, 13463, 13464, 13465, 13467, 13470, 13475, 13476, 13477, 13479, 13480, 13484, 13487, 13495, 13496, 13497, 13501, 13503, 13505, 13506, 13509, 13510, 13513, 13515, 13516, 13517, 13518, 13520, 13521, 13522, 13525, 13527, 13528, 13533, 13539, 13540, 13544, 13545, 13549, 13550, 13555, 13556, 13559, 13565, 13566, 13568, 13571, 13572, 13573, 13574, 13577, 13580, 13581, 13584, 13589, 13590, 13592, 13597, 13598, 13601, 13605, 13609, 13610, 13614, 13619, 13621, 13623, 13624, 13626, 13627, 13628, 13632, 13633, 13636, 13639, 13641, 13643, 13644, 13645, 13647, 13650, 13653, 13656, 13661, 13662, 13663, 13675, 13676, 13677, 13678, 13680, 13685, 13687, 13690, 13694, 13699, 13700, 13702, 13704, 13705, 13708, 13717, 13721, 13722, 13733, 13735, 13736, 13738, 13741, 13743, 13748, 13750, 13758, 13761, 13763, 13765, 13766, 13769, 13771, 13773, 13776, 13778, 13779, 13791, 13794, 13798, 13800, 13803, 13804, 13812, 13816, 13821, 13822, 13829, 13831, 13832, 13841, 13842, 13846, 13848, 13851, 13855, 13865, 13866, 13871, 13874, 13875, 13878, 13880, 13882, 13886, 13887, 13889, 13890, 13891, 13892, 13894, 13897, 13898, 13900, 13902, 13903, 13907, 13912, 13915, 13916, 13919, 13921, 13924, 13926, 13927, 13930, 13932, 13933, 13934, 13936, 13939, 13943, 13944, 13945, 13946, 13948, 13952, 13953, 13955, 13959, 13962, 13966, 13967, 13968, 13969, 13970, 13972, 13973, 13974, 13976, 13980, 13981, 13982, 13983, 13988, 13989, 13991, 13993, 13994, 13996, 14001, 14003, 14004, 14006, 14013, 14015, 14017, 14018, 14019, 14021, 14023, 14026, 14029, 14038, 14040, 14050, 14053, 14059, 14060, 14063, 14067, 14069, 14074, 14076, 14077, 14078, 14080, 14081, 14082, 14083, 14084, 14085, 14087, 14090, 14091, 14095, 14099, 14105, 14107, 14108, 14111, 14112, 14114, 14115, 14117, 14118, 14119, 14121, 14122, 14123, 14125, 14135, 14136, 14140, 14141, 14143, 14146, 14147, 14154, 14160, 14164, 14168, 14169, 14172, 14174, 14175, 14176, 14181, 14182, 14188, 14193, 14198, 14200, 14202, 14211, 14212, 14213, 14214, 14215, 14220, 14225, 14226, 14237, 14239, 14240, 14242, 14246, 14248, 14249, 14250, 14255, 14257, 14266, 14267, 14269, 14270, 14271, 14272, 14281, 14282, 14289, 14292, 14295, 14298, 14299, 14300, 14301, 14302, 14303, 14308, 14310, 14311, 14312, 14316, 14318, 14320, 14321, 14324, 14327, 14332, 14336, 14338, 14339, 14349, 14352, 14355, 14357, 14360, 14361, 14362, 14363, 14371, 14372, 14373, 14376, 14377, 14381, 14383, 14384, 14385, 14386, 14389, 14393, 14396, 14403, 14404, 14406, 14410, 14412, 14413, 14416, 14417, 14418, 14422, 14428, 14431, 14433, 14435, 14437, 14441, 14446, 14447, 14450, 14453, 14454, 14458, 14459, 14460, 14466, 14467, 14468, 14469, 14478, 14481, 14483, 14485, 14488, 14491, 14492, 14493, 14494, 14495, 14496, 14497, 14499, 14500, 14501, 14511, 14512, 14514, 14516, 14517, 14519, 14523, 14526, 14527, 14528, 14532, 14543, 14546, 14548, 14552, 14555, 14556, 14558, 14559, 14564, 14565, 14566, 14573, 14575, 14581, 14586, 14587, 14592, 14594, 14595, 14599, 14603, 14604, 14608, 14612, 14614, 14617, 14633, 14637, 14639, 14641, 14643, 14644, 14652, 14654, 14656, 14663, 14667, 14669, 14671, 14677, 14681, 14685, 14688, 14689, 14694, 14699, 14704, 14705, 14709, 14712, 14719, 14720, 14721, 14724, 14725, 14728, 14730, 14737, 14738, 14740, 14742, 14747, 14752, 14754, 14755, 14757, 14758, 14759, 14761, 14764, 14768, 14771, 14774, 14778, 14779, 14782, 14783, 14787, 14788, 14793, 14796, 14797, 14798, 14801, 14802, 14804, 14805, 14816, 14817, 14819, 14826, 14830, 14831, 14833, 14836, 14837, 14839, 14840, 14847, 14848, 14849, 14850, 14864, 14867, 14868, 14869, 14873, 14874, 14875, 14879, 14880, 14881, 14882, 14883, 14886, 14892, 14893, 14895, 14898, 14903, 14904, 14910, 14916, 14918, 14919, 14921, 14923, 14924, 14934, 14937, 14938, 14940, 14942, 14947, 14949, 14950, 14951, 14953, 14954, 14955, 14956, 14958, 14963, 14965, 14976, 14978, 14979, 14981, 14982, 14983, 14984, 14990, 14992, 14996, 14997, 14999, 15003, 15008, 15009, 15019, 15020, 15022, 15026, 15035, 15037, 15038, 15039, 15045, 15046, 15053, 15057, 15058, 15063, 15065, 15067, 15068, 15074, 15078, 15080, 15081, 15082, 15083, 15084, 15085, 15091, 15093, 15096, 15097, 15104, 15105, 15109, 15111, 15114, 15118, 15127, 15130, 15136, 15138, 15141, 15145, 15146, 15147, 15149, 15151, 15154, 15155, 15164, 15167, 15169, 15177, 15184, 15186, 15187, 15188, 15190, 15198, 15199, 15203, 15207, 15214, 15215, 15221, 15223, 15226, 15229, 15232, 15237, 15239, 15241, 15245, 15256, 15258, 15259, 15261, 15263, 15264, 15266, 15268, 15270, 15271, 15278, 15279, 15281, 15288, 15289, 15292, 15300, 15301, 15302, 15303, 15308, 15311, 15314, 15315, 15316, 15318, 15322, 15326, 15331, 15333, 15336, 15338, 15340, 15342, 15345, 15346, 15348, 15350, 15360, 15361, 15366, 15367, 15368, 15369, 15370, 15372, 15373, 15375, 15376, 15377, 15378, 15387, 15393, 15394, 15395, 15396, 15397, 15398, 15400, 15403, 15405, 15407, 15410, 15411, 15412, 15414, 15416, 15417, 15419, 15422, 15423, 15425, 15427, 15429, 15433, 15436, 15437, 15438, 15444, 15445, 15451, 15454, 15457, 15458, 15459, 15461, 15462, 15469, 15471, 15472, 15477, 15480, 15481, 15482, 15483, 15485, 15487, 15496, 15502, 15503, 15506, 15507, 15522, 15525, 15526, 15532, 15533, 15534, 15537, 15542, 15546, 15549, 15550, 15551, 15554, 15560, 15562, 15564, 15565, 15566, 15570, 15571, 15572, 15574, 15579, 15580, 15585, 15587, 15597, 15608, 15609, 15618, 15620, 15621, 15622, 15630, 15632, 15633, 15637, 15638, 15644, 15645, 15646, 15647, 15652, 15653, 15654, 15656, 15662, 15663, 15665, 15666, 15671, 15676, 15680, 15681, 15685, 15687, 15689, 15696, 15698, 15699, 15701, 15702, 15703, 15708, 15710, 15714, 15716, 15718, 15723, 15726, 15727, 15732, 15735, 15737, 15739, 15748, 15751, 15754, 15755, 15763, 15768, 15770, 15776, 15777, 15781, 15786, 15787, 15788, 15792, 15793, 15799, 15805, 15807, 15808, 15812, 15813, 15822, 15827, 15832, 15837, 15838, 15840, 15848, 15850, 15851, 15853, 15854, 15857, 15858, 15860, 15864, 15868, 15869, 15870, 15872, 15874, 15877, 15882, 15886, 15889, 15892, 15893, 15895, 15898, 15899, 15900, 15901, 15904, 15907, 15911, 15912, 15914, 15915, 15916, 15917, 15918, 15920, 15921, 15936, 15941, 15942, 15944, 15948, 15951, 15957, 15958, 15962, 15965, 15971, 15972, 15973, 15974, 15976, 15978, 15980, 15981, 15986, 15989, 15990, 15991, 15995, 15997, 16002, 16003, 16006, 16015, 16017, 16019, 16021, 16022, 16023, 16025, 16028, 16029, 16030, 16032, 16033, 16034, 16035, 16038, 16040, 16041, 16042, 16045, 16046, 16047, 16052, 16053, 16058, 16060, 16064, 16073, 16075, 16077, 16082, 16083, 16084, 16085, 16089, 16092, 16094, 16095, 16096, 16097, 16099, 16101]


def get_in_lex_sentences(out_lex_idxs, src_file):
    idxs = [i for i in range(0, 16102)]
    in_lex_ids = [x for x in idxs if x not in out_lex_idxs]

    clean_ids = get_non_ascii_lines(src_file)

    in_lex_clean_ids = [x for x in in_lex_ids if x in clean_ids]

    in_lex_train_ids = [x for x in in_lex_clean_ids if x < 14486]
    clean_eval_ids = [x for x in clean_ids if x > 14486]

    # assert len(in_lex_train_ids) == 9207
    # assert len(clean_eval_ids) == 1594

    return in_lex_train_ids, clean_eval_ids


# get_in_lex_sentences(ood_sentence_idxs, "WHD_data/data/WHD_src.txt")


def get_WHD_training_eval_data(out_lex_ids, src_file, tgt_file, POS_file):
    in_lex_train_ids, clean_eval_ids = get_in_lex_sentences(out_lex_ids, src_file + ".txt")

    with open(src_file + ".txt", "r") as src:
        with open(tgt_file + ".txt", "r") as tgt:
            with open(POS_file + ".txt", "r") as POS:
                src_lines = src.readlines()
                tgt_lines = tgt.readlines()
                POS_lines = POS.readlines()

    assert len(src_lines) == len(tgt_lines) == len(POS_lines)

    src_train = [src for i, src in enumerate(src_lines) if i in in_lex_train_ids]
    tgt_train = [tgt for i, tgt in enumerate(tgt_lines) if i in in_lex_train_ids]
    POS_train = [POS for i, POS in enumerate(POS_lines) if i in in_lex_train_ids]

    assert len(src_train) == len(tgt_train) == len(POS_train)

    src_eval = [src for i, src in enumerate(src_lines) if i in clean_eval_ids]
    tgt_eval = [tgt for i, tgt in enumerate(tgt_lines) if i in clean_eval_ids]
    POS_eval = [POS for i, POS in enumerate(POS_lines) if i in clean_eval_ids]

    assert len(src_eval) == len(tgt_eval) == len(POS_eval)

    # with open(src_file + "_train.txt", "w") as src:
    #     with open(tgt_file + "_train.txt", "w") as tgt:
    #         with open(POS_file + "_train.txt", "w") as POS:
    #             for src_line, tgt_line, POS_line in zip(src_train, tgt_train, POS_train):
    #                 src.write(src_line)
    #                 tgt.write(tgt_line)
    #                 POS.write(POS_line)
    #
    # with open(src_file + "_eval.txt", "w") as src:
    #     with open(tgt_file + "_eval.txt", "w") as tgt:
    #         with open(POS_file + "_eval.txt", "w") as POS:
    #             for src_line, tgt_line, POS_line in zip(src_eval, tgt_eval, POS_eval):
    #                 src.write(src_line)
    #                 tgt.write(tgt_line)
    #                 POS.write(POS_line)

    return src_train, tgt_train, POS_train, src_eval, tgt_eval, POS_eval


# get_WHD_training_eval_data(ood_sentence_idxs, "WHD_data/data/WHD_src", "WHD_data/data/WHD_tgt", "WHD_data/data/WHD_POS")


def get_WHD_training_eval_df(out_lex_ids, WHD_full, src_file):
    in_lex_train_ids, clean_eval_ids = get_in_lex_sentences(out_lex_ids, src_file + ".txt")

    with open(WHD_full, "rb") as df:
        WHD = pickle.load(df)

    WHD_train_inlex = WHD.iloc[in_lex_train_ids]
    WHD_eval_clean = WHD.iloc[clean_eval_ids]

    # with open("WHD_train_inlex_NEW.pkl", "wb") as f:
    #     with open("WHD_eval_clean_NEW.pkl", "wb") as f2:
    #         pickle.dump(WHD_train_inlex, f)
    #         pickle.dump(WHD_eval_clean, f2)
    #
    # with open("WHD_train_inlex_NEW.pkl", "rb") as f:
    #     with open("WHD_eval_clean_NEW.pkl", "rb") as f2:
    #         WHD_train = pickle.load(f)
    #         WHD_eval = pickle.load(f2)

    # print(WHD_train.info())
    # print(WHD_eval.info())

# get_WHD_training_eval_df(ood_sentence_idxs, "WHD_full.pkl", "WHD_data/data/WHD_src")


def get_homograph_dist(WHD_df, homograph_outfile, wordid_outfile):
    with open(WHD_df, "rb") as df:
        WHD = pickle.load(df)

    homographs = WHD['homograph'].tolist()
    wordids = WHD['wordid'].tolist()

    wordid_count = defaultdict(lambda: 0)
    for wid in wordids:
        wordid_count[wid] += 1

    homograph_count = defaultdict(lambda: 0)
    for homograph in homographs:
        homograph_count[homograph] += 1

    homographs = list(set(homographs))
    wordids = list(set(wordids))
    print("no. of homographs: ", len(homographs))
    print("no. of wordids: ", len(wordids))

    with open(homograph_outfile, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in homograph_count.items():
            writer.writerow([key, value])

    with open(wordid_outfile, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in wordid_count.items():
            writer.writerow([key, value])


# get_homograph_dist("WHD_eval.pkl", "WHD_data/eval_homograph_counts.csv", "WHD_data/eval_wordid_counts.csv")







