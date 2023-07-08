import pickle
import pandas as pd
import csv
import re
from collections import defaultdict
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


def get_homograph_pron_stats(phonetic_seqs, df):
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
                # matches = []
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

    # with open("festival_analysis/pronunciation_incorrect.csv", 'w') as f:
    #     # using csv.writer method from CSV package
    #     write = csv.writer(f)
    #     write.writerows(bad_preds)
    #
    # with open("festival_analysis/just_pronunciation_correct.csv", 'w') as f:
    #     # using csv.writer method from CSV package
    #     write = csv.writer(f)
    #     write.writerows(correct_prons)
    #
    # with open("festival_analysis/fully_correct.csv", "w") as f:
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
        if i in idxs:
            if len(word_seq) == len(pred_seq):
                homograph_idx = word_seq.index(homograph)
                pred_seq[homograph_idx] = (gt[1], gt[2])
                corrected_preds.append(pred_seq)
            else:
                potential_preds = homograph_dict[homograph]
                for pot_pred in potential_preds:
                    pot_pred = (pot_pred[1], pot_pred[2])
                    if pot_pred in pred_seq:
                        homograph_idx = pred_seq.index(pot_pred)
                        pred_seq[homograph_idx] = (gt[1], gt[2])
                        corrected_preds.append(pred_seq)
                        break
                # print("\n", pred_seq)
                # print(homograph)
                # homograph_idx = word_seq.index(homograph)
                # print("proposed idx: ", homograph_idx)
                # try:
                #     print("word to change: ", pred_seq[homograph_idx])
                # except IndexError:
                #     print("word to change (final idx): ", pred_seq[-1])
                # validation = input("ok?")
                # if validation == "y":
                #     pred_seq[homograph_idx] = (gt[1], gt[2])
                #     corrected_preds.append((i, pred_seq))
                #     write.writerow(corrected_preds[-1])
                # if validation == "n":
                #     print((gt[1], gt[2]))
                #     corrected_idx = int(input("corrected pred_seq idx: "))
                #     print("word to change: ", pred_seq[corrected_idx])
                #     pred_seq[corrected_idx] = (gt[1], gt[2])
                #     corrected_preds.append((i, pred_seq))
                #     write.writerow(corrected_preds[-1])
        else:
            corrected_preds.append(pred_seq)


    assert len(corrected_preds) == 16102

    with open("festival_analysis/corrected_preds.csv", 'w') as word:
        # using csv.writer method from CSV package
        write = csv.writer(word)
        write.writerows(corrected_preds)

    return corrected_preds


get_homograph_pron_stats("WHD_data/data/WHD_corrected_pron.txt", "WHD_full.pkl")


def find_indices(search_list, search_item):
    return [index for (index, item) in enumerate(search_list) if item == search_item]


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


# postprocess_preds(correct_preds("WHD_data/data/WHD_pron.txt", "WHD_full.pkl", get_lex_homograph_prons_dict("unilex-rpx.out", "WHD_full.pkl")), "WHD_data/data/WHD_pron.txt", "WHD_data/data/WHD_corrected_pron.txt")


def get_gt_homograph_prons(df, homograph_lex):
    with open(df, "rb") as df:
        WHD = pickle.load(df)

    homograph_lex = homograph_lex

    word_ids = WHD['wordid'].tolist()
    homographs = WHD['homograph'].tolist()

    # get corresponding word_ids and homograph lists
    set_word_ids = list(set(word_ids))

    corresponding_homographs = []
    for id in set_word_ids:
        first_appearance_idx = find_indices(word_ids, id)[0]
        corresponding_homographs.append(homographs[first_appearance_idx])

    assert len(set_word_ids) == len(corresponding_homographs)

    gt_pronunciations = []
    i = 0
    for wordid, homograph in zip(set_word_ids, corresponding_homographs):
        i += 1
        potential_prons = homograph_lex[homograph]
        print(i, "\t", wordid, "\t", homograph, "\t", potential_prons)
        chosen_pron_idx = int(input("Chosen pron idx: "))
        gt_pronunciations.append(potential_prons[chosen_pron_idx])
        print(gt_pronunciations)

    print(gt_pronunciations)
    input("OK?")
    assert len(gt_pronunciations) == len(set_word_ids)

    zipped_id_pron = list(zip(set_word_ids, gt_pronunciations))
    df = pd.DataFrame(zipped_id_pron, columns=['wordid', 'gt_homograph_pron'])

    WHD = WHD.merge(df, on='wordid', how='left')
    print(WHD.info())
    print(WHD.head().to_string())
    input("OK?")

    with open(f"WHD_FULL.pkl", "wb") as out_df:
        pickle.dump(WHD, out_df)


out_lex_idxs = [0, 3, 3, 4, 4, 4, 6, 10, 10, 11, 15, 18, 18, 19, 22, 22, 24, 25, 26, 28, 32, 38, 39, 39, 39, 39, 41, 42, 42, 44, 45, 47, 48, 48, 48, 53, 55, 57, 60, 63, 63, 68, 69, 69, 71, 74, 75, 76, 77, 77, 78, 79, 82, 83, 85, 86, 86, 87, 87, 88, 88, 91, 91, 93, 93, 94, 95, 95, 96, 96, 99, 99, 100, 107, 107, 109, 110, 115, 115, 116, 117, 119, 120, 121, 122, 123, 128, 128, 130, 132, 137, 138, 138, 138, 138, 138, 138, 138, 139, 142, 142, 143, 144, 148, 148, 149, 151, 151, 152, 152, 152, 153, 153, 154, 154, 156, 156, 157, 159, 160, 163, 164, 164, 164, 164, 166, 166, 167, 168, 169, 170, 170, 173, 179, 181, 182, 182, 182, 184, 184, 184, 187, 187, 187, 189, 190, 193, 193, 194, 195, 196, 196, 200, 200, 202, 202, 204, 204, 206, 206, 207, 209, 209, 214, 219, 219, 221, 223, 223, 223, 223, 225, 225, 226, 229, 235, 241, 241, 241, 242, 245, 248, 250, 250, 251, 251, 251, 251, 252, 255, 255, 255, 255, 260, 262, 262, 262, 263, 264, 264, 265, 265, 265, 266, 266, 266, 266, 267, 268, 269, 270, 270, 270, 272, 273, 273, 275, 280, 282, 284, 284, 285, 285, 287, 290, 292, 295, 299, 299, 299, 299, 301, 302, 308, 308, 317, 318, 318, 318, 318, 319, 321, 323, 327, 329, 329, 331, 331, 333, 336, 337, 337, 339, 341, 342, 342, 343, 343, 343, 346, 347, 350, 351, 351, 351, 351, 356, 358, 361, 361, 363, 363, 364, 366, 367, 369, 371, 371, 375, 375, 375, 379, 382, 383, 385, 386, 388, 388, 389, 390, 391, 391, 391, 391, 392, 394, 394, 396, 400, 400, 400, 400, 404, 405, 406, 406, 407, 412, 414, 415, 415, 416, 419, 419, 420, 421, 421, 421, 421, 422, 422, 424, 425, 425, 428, 429, 430, 430, 434, 436, 440, 442, 442, 443, 443, 444, 446, 446, 447, 448, 448, 449, 453, 457, 458, 458, 460, 461, 462, 462, 464, 465, 465, 465, 465, 465, 465, 470, 472, 472, 474, 475, 475, 477, 478, 480, 482, 483, 483, 483, 483, 484, 485, 485, 487, 488, 488, 488, 489, 489, 489, 490, 491, 491, 491, 493, 493, 493, 493, 493, 495, 497, 499, 500, 501, 503, 506, 506, 506, 506, 507, 507, 509, 510, 510, 511, 512, 514, 515, 515, 517, 517, 519, 520, 520, 522, 522, 524, 525, 525, 529, 529, 533, 533, 533, 533, 534, 536, 537, 537, 537, 538, 538, 539, 540, 545, 545, 546, 546, 547, 549, 552, 553, 554, 555, 558, 560, 562, 562, 563, 563, 571, 574, 576, 576, 577, 578, 582, 585, 594, 594, 595, 598, 599, 599, 599, 603, 608, 610, 613, 615, 615, 615, 618, 622, 623, 625, 629, 630, 630, 631, 632, 632, 633, 633, 633, 633, 636, 636, 638, 638, 639, 639, 641, 645, 645, 645, 645, 645, 653, 657, 659, 664, 668, 668, 669, 669, 670, 673, 676, 678, 681, 681, 681, 683, 688, 689, 691, 692, 694, 695, 697, 697, 700, 701, 703, 703, 705, 707, 708, 710, 710, 711, 715, 716, 717, 718, 719, 721, 722, 726, 730, 732, 732, 733, 734, 734, 736, 738, 739, 740, 740, 742, 743, 744, 745, 746, 747, 752, 753, 755, 755, 760, 761, 761, 762, 762, 763, 763, 766, 766, 770, 773, 774, 777, 779, 781, 781, 781, 782, 783, 783, 784, 784, 785, 785, 785, 788, 788, 788, 788, 788, 790, 790, 793, 793, 794, 795, 798, 807, 811, 811, 812, 812, 812, 814, 816, 817, 819, 823, 825, 825, 826, 826, 830, 833, 834, 837, 837, 838, 838, 842, 842, 844, 850, 850, 852, 853, 855, 857, 858, 863, 864, 864, 864, 865, 869, 869, 869, 871, 871, 872, 874, 876, 877, 878, 878, 883, 885, 888, 891, 893, 894, 895, 896, 901, 901, 902, 902, 903, 903, 905, 905, 909, 910, 915, 916, 917, 922, 922, 922, 923, 924, 929, 929, 931, 933, 935, 935, 935, 935, 935, 939, 940, 941, 942, 942, 943, 943, 944, 944, 944, 945, 947, 947, 949, 952, 953, 955, 959, 960, 963, 964, 964, 966, 968, 969, 976, 977, 977, 979, 979, 979, 980, 982, 983, 983, 984, 986, 986, 988, 990, 991, 998, 1001, 1002, 1003, 1003, 1004, 1004, 1007, 1008, 1012, 1013, 1015, 1018, 1020, 1021, 1024, 1028, 1028, 1029, 1030, 1030, 1030, 1031, 1032, 1035, 1036, 1040, 1042, 1042, 1044, 1044, 1044, 1046, 1046, 1046, 1048, 1050, 1051, 1057, 1058, 1059, 1061, 1069, 1073, 1075, 1076, 1080, 1082, 1084, 1093, 1094, 1098, 1103, 1104, 1105, 1105, 1107, 1107, 1108, 1108, 1108, 1109, 1110, 1111, 1114, 1115, 1117, 1117, 1119, 1120, 1121, 1124, 1127, 1127, 1129, 1129, 1129, 1134, 1135, 1141, 1143, 1143, 1144, 1146, 1152, 1152, 1154, 1155, 1155, 1156, 1157, 1159, 1159, 1160, 1161, 1161, 1162, 1168, 1168, 1171, 1177, 1177, 1179, 1179, 1180, 1181, 1181, 1182, 1183, 1184, 1184, 1185, 1185, 1187, 1188, 1188, 1189, 1192, 1194, 1198, 1199, 1200, 1204, 1204, 1206, 1208, 1208, 1208, 1211, 1211, 1211, 1211, 1212, 1213, 1214, 1216, 1216, 1217, 1220, 1220, 1223, 1226, 1226, 1226, 1226, 1227, 1229, 1235, 1238, 1242, 1242, 1248, 1248, 1248, 1248, 1249, 1254, 1255, 1256, 1257, 1262, 1263, 1266, 1266, 1268, 1271, 1272, 1273, 1273, 1273, 1275, 1275, 1276, 1278, 1279, 1279, 1282, 1288, 1289, 1297, 1300, 1300, 1301, 1303, 1304, 1305, 1308, 1312, 1313, 1313, 1315, 1315, 1316, 1317, 1317, 1325, 1325, 1326, 1327, 1327, 1327, 1331, 1332, 1333, 1333, 1334, 1337, 1339, 1341, 1342, 1344, 1344, 1346, 1346, 1347, 1349, 1352, 1352, 1352, 1352, 1352, 1352, 1353, 1355, 1356, 1357, 1359, 1360, 1361, 1361, 1363, 1366, 1367, 1367, 1368, 1371, 1371, 1373, 1375, 1377, 1377, 1378, 1382, 1382, 1386, 1386, 1386, 1387, 1387, 1387, 1389, 1390, 1391, 1392, 1395, 1395, 1395, 1395, 1397, 1397, 1400, 1400, 1401, 1402, 1404, 1404, 1405, 1407, 1407, 1408, 1410, 1411, 1412, 1412, 1415, 1415, 1415, 1416, 1416, 1417, 1417, 1418, 1418, 1420, 1421, 1424, 1425, 1425, 1426, 1427, 1429, 1429, 1430, 1433, 1436, 1436, 1436, 1439, 1439, 1439, 1439, 1440, 1441, 1441, 1441, 1441, 1441, 1445, 1445, 1445, 1446, 1449, 1449, 1452, 1453, 1453, 1455, 1456, 1457, 1457, 1457, 1461, 1461, 1466, 1468, 1468, 1470, 1471, 1473, 1474, 1474, 1474, 1476, 1476, 1476, 1481, 1481, 1482, 1486, 1492, 1493, 1493, 1493, 1494, 1494, 1494, 1497, 1499, 1501, 1501, 1503, 1503, 1503, 1504, 1505, 1508, 1513, 1514, 1516, 1517, 1518, 1520, 1522, 1522, 1527, 1527, 1527, 1530, 1530, 1543, 1544, 1549, 1549, 1550, 1550, 1552, 1554, 1556, 1559, 1560, 1564, 1565, 1566, 1568, 1569, 1575, 1576, 1576, 1580, 1580, 1580, 1583, 1583, 1586, 1586, 1587, 1587, 1589, 1591, 1591, 1593, 1593, 1594, 1596, 1598, 1598, 1601, 1603, 1605, 1610, 1611, 1611, 1619, 1621, 1621, 1622, 1622, 1623, 1624, 1626, 1626, 1626, 1629, 1630, 1631, 1632, 1634, 1636, 1639, 1642, 1643, 1643, 1643, 1645, 1645, 1646, 1650, 1652, 1655, 1661, 1671, 1674, 1683, 1683, 1686, 1686, 1686, 1688, 1690, 1694, 1701, 1704, 1705, 1708, 1709, 1710, 1710, 1710, 1710, 1711, 1713, 1714, 1714, 1721, 1726, 1730, 1734, 1734, 1735, 1736, 1739, 1739, 1739, 1739, 1739, 1744, 1745, 1750, 1752, 1752, 1753, 1757, 1759, 1762, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1780, 1783, 1783, 1783, 1783, 1785, 1785, 1789, 1789, 1792, 1793, 1794, 1795, 1795, 1796, 1797, 1797, 1801, 1804, 1804, 1804, 1805, 1808, 1808, 1810, 1810, 1810, 1810, 1811, 1811, 1813, 1814, 1814, 1815, 1816, 1819, 1820, 1821, 1822, 1822, 1823, 1824, 1824, 1825, 1825, 1826, 1829, 1831, 1832, 1832, 1833, 1834, 1835, 1841, 1842, 1842, 1843, 1843, 1844, 1844, 1846, 1846, 1846, 1847, 1850, 1851, 1852, 1853, 1854, 1856, 1856, 1858, 1859, 1860, 1861, 1861, 1862, 1863, 1863, 1863, 1863, 1863, 1865, 1865, 1867, 1868, 1870, 1872, 1872, 1872, 1872, 1872, 1872, 1874, 1874, 1877, 1877, 1880, 1880, 1880, 1880, 1881, 1883, 1885, 1889, 1889, 1897, 1897, 1898, 1898, 1902, 1902, 1902, 1903, 1904, 1906, 1906, 1907, 1909, 1911, 1911, 1913, 1920, 1926, 1928, 1930, 1930, 1932, 1937, 1939, 1939, 1942, 1942, 1943, 1949, 1949, 1952, 1952, 1953, 1953, 1954, 1955, 1955, 1956, 1957, 1958, 1958, 1958, 1958, 1958, 1959, 1963, 1964, 1966, 1967, 1971, 1972, 1974, 1974, 1980, 1984, 1986, 1990, 1995, 1999, 2003, 2003, 2003, 2004, 2008, 2008, 2014, 2014, 2015, 2018, 2018, 2021, 2021, 2021, 2023, 2023, 2023, 2026, 2029, 2029, 2033, 2033, 2033, 2033, 2034, 2034, 2035, 2035, 2039, 2039, 2039, 2042, 2043, 2046, 2046, 2047, 2047, 2052, 2058, 2059, 2061, 2062, 2062, 2067, 2069, 2075, 2075, 2077, 2078, 2078, 2079, 2079, 2079, 2079, 2082, 2083, 2083, 2086, 2086, 2086, 2087, 2088, 2090, 2090, 2092, 2092, 2092, 2093, 2093, 2094, 2095, 2095, 2096, 2097, 2098, 2098, 2099, 2099, 2100, 2102, 2104, 2106, 2107, 2108, 2109, 2109, 2112, 2117, 2117, 2117, 2118, 2118, 2118, 2118, 2119, 2119, 2120, 2121, 2121, 2123, 2124, 2125, 2126, 2132, 2132, 2134, 2137, 2137, 2137, 2139, 2139, 2140, 2141, 2141, 2149, 2150, 2151, 2152, 2152, 2153, 2153, 2154, 2155, 2157, 2158, 2161, 2164, 2170, 2171, 2172, 2176, 2176, 2176, 2177, 2185, 2188, 2189, 2190, 2191, 2192, 2194, 2195, 2196, 2196, 2196, 2198, 2199, 2199, 2202, 2203, 2204, 2204, 2205, 2205, 2205, 2205, 2210, 2212, 2214, 2216, 2226, 2228, 2230, 2231, 2232, 2233, 2235, 2243, 2247, 2249, 2250, 2251, 2251, 2252, 2256, 2256, 2256, 2257, 2257, 2257, 2257, 2261, 2264, 2264, 2266, 2267, 2268, 2268, 2269, 2275, 2275, 2275, 2278, 2279, 2279, 2279, 2279, 2283, 2283, 2283, 2285, 2285, 2286, 2287, 2288, 2288, 2290, 2291, 2291, 2291, 2295, 2298, 2298, 2299, 2302, 2303, 2303, 2306, 2306, 2306, 2306, 2307, 2309, 2310, 2311, 2313, 2315, 2318, 2320, 2323, 2323, 2323, 2326, 2326, 2330, 2331, 2332, 2334, 2338, 2339, 2339, 2339, 2343, 2345, 2348, 2350, 2351, 2352, 2353, 2354, 2354, 2355, 2356, 2357, 2357, 2357, 2357, 2357, 2358, 2360, 2363, 2364, 2368, 2368, 2368, 2370, 2370, 2370, 2371, 2372, 2372, 2372, 2373, 2374, 2374, 2376, 2377, 2377, 2377, 2378, 2380, 2381, 2381, 2382, 2382, 2385, 2385, 2388, 2388, 2389, 2392, 2392, 2393, 2393, 2393, 2394, 2397, 2398, 2399, 2400, 2405, 2410, 2416, 2416, 2422, 2423, 2424, 2424, 2428, 2433, 2433, 2434, 2434, 2435, 2435, 2435, 2436, 2436, 2437, 2438, 2439, 2441, 2441, 2442, 2442, 2443, 2443, 2445, 2446, 2449, 2449, 2450, 2450, 2451, 2451, 2451, 2451, 2452, 2453, 2453, 2454, 2455, 2459, 2460, 2464, 2465, 2466, 2468, 2470, 2470, 2472, 2473, 2474, 2474, 2474, 2479, 2480, 2480, 2481, 2482, 2482, 2483, 2485, 2486, 2486, 2487, 2487, 2487, 2487, 2488, 2488, 2493, 2493, 2495, 2497, 2499, 2499, 2499, 2500, 2507, 2508, 2509, 2509, 2510, 2515, 2519, 2520, 2520, 2520, 2523, 2523, 2523, 2524, 2527, 2529, 2529, 2534, 2534, 2534, 2535, 2535, 2536, 2536, 2539, 2539, 2540, 2543, 2543, 2545, 2553, 2554, 2556, 2556, 2557, 2559, 2561, 2565, 2566, 2567, 2567, 2570, 2572, 2573, 2574, 2574, 2575, 2577, 2578, 2580, 2580, 2581, 2589, 2590, 2591, 2591, 2591, 2592, 2592, 2593, 2594, 2594, 2594, 2595, 2595, 2598, 2598, 2602, 2603, 2604, 2604, 2604, 2605, 2605, 2606, 2606, 2607, 2607, 2609, 2615, 2615, 2616, 2620, 2620, 2623, 2623, 2623, 2623, 2624, 2624, 2624, 2624, 2624, 2624, 2626, 2627, 2630, 2633, 2633, 2636, 2636, 2641, 2642, 2642, 2644, 2646, 2648, 2650, 2655, 2655, 2657, 2658, 2658, 2658, 2659, 2662, 2663, 2663, 2664, 2667, 2668, 2668, 2669, 2669, 2673, 2673, 2674, 2683, 2684, 2684, 2686, 2688, 2690, 2693, 2698, 2699, 2701, 2702, 2703, 2706, 2707, 2707, 2707, 2709, 2710, 2712, 2713, 2713, 2714, 2715, 2716, 2717, 2720, 2728, 2729, 2730, 2732, 2736, 2737, 2737, 2738, 2741, 2744, 2749, 2753, 2755, 2755, 2756, 2761, 2762, 2762, 2763, 2763, 2766, 2766, 2768, 2768, 2770, 2772, 2773, 2775, 2775, 2777, 2780, 2780, 2780, 2783, 2789, 2790, 2792, 2793, 2793, 2795, 2796, 2797, 2797, 2801, 2805, 2807, 2807, 2808, 2808, 2808, 2810, 2810, 2813, 2815, 2815, 2816, 2821, 2821, 2824, 2824, 2826, 2835, 2836, 2836, 2836, 2840, 2841, 2844, 2844, 2845, 2850, 2853, 2857, 2860, 2861, 2862, 2863, 2865, 2865, 2870, 2871, 2872, 2872, 2872, 2874, 2876, 2876, 2878, 2880, 2881, 2881, 2882, 2885, 2893, 2904, 2905, 2907, 2907, 2908, 2911, 2912, 2913, 2914, 2915, 2917, 2918, 2918, 2919, 2920, 2921, 2924, 2926, 2926, 2927, 2929, 2929, 2930, 2940, 2942, 2946, 2949, 2952, 2952, 2952, 2953, 2953, 2955, 2956, 2956, 2957, 2957, 2957, 2957, 2957, 2958, 2959, 2960, 2961, 2962, 2964, 2964, 2964, 2965, 2967, 2969, 2971, 2971, 2972, 2973, 2974, 2975, 2976, 2976, 2982, 2983, 2983, 2985, 2987, 2989, 2990, 2991, 2992, 2993, 2993, 2996, 2996, 2996, 2996, 2998, 2998, 2999, 2999, 2999, 3003, 3006, 3009, 3010, 3010, 3010, 3012, 3015, 3015, 3015, 3017, 3017, 3017, 3018, 3020, 3021, 3022, 3025, 3025, 3026, 3028, 3028, 3028, 3029, 3031, 3032, 3033, 3033, 3034, 3035, 3035, 3035, 3036, 3036, 3036, 3037, 3037, 3044, 3044, 3044, 3045, 3047, 3047, 3047, 3051, 3055, 3055, 3055, 3057, 3057, 3057, 3057, 3057, 3057, 3058, 3059, 3060, 3061, 3062, 3062, 3063, 3065, 3067, 3068, 3070, 3070, 3074, 3074, 3075, 3076, 3076, 3078, 3079, 3079, 3081, 3083, 3083, 3084, 3084, 3085, 3085, 3086, 3086, 3087, 3087, 3095, 3095, 3096, 3097, 3099, 3099, 3099, 3099, 3100, 3100, 3101, 3102, 3103, 3104, 3109, 3109, 3109, 3111, 3112, 3112, 3112, 3112, 3112, 3112, 3112, 3113, 3113, 3114, 3116, 3116, 3117, 3124, 3125, 3125, 3125, 3125, 3126, 3128, 3135, 3137, 3137, 3137, 3140, 3140, 3140, 3141, 3143, 3143, 3144, 3145, 3148, 3148, 3148, 3148, 3149, 3149, 3154, 3154, 3154, 3156, 3156, 3156, 3156, 3157, 3159, 3160, 3162, 3163, 3164, 3164, 3167, 3171, 3171, 3172, 3172, 3174, 3175, 3175, 3177, 3177, 3180, 3180, 3182, 3182, 3182, 3183, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3192, 3192, 3193, 3194, 3194, 3195, 3195, 3197, 3197, 3197, 3197, 3198, 3199, 3199, 3199, 3199, 3199, 3199, 3200, 3200, 3201, 3202, 3203, 3204, 3204, 3206, 3208, 3210, 3213, 3213, 3214, 3216, 3216, 3218, 3218, 3218, 3221, 3222, 3222, 3223, 3224, 3226, 3227, 3228, 3228, 3228, 3231, 3234, 3234, 3234, 3234, 3235, 3239, 3240, 3240, 3241, 3243, 3247, 3248, 3248, 3249, 3249, 3251, 3252, 3252, 3253, 3254, 3254, 3255, 3255, 3255, 3260, 3260, 3265, 3265, 3266, 3267, 3268, 3268, 3269, 3269, 3272, 3272, 3273, 3273, 3275, 3275, 3276, 3276, 3278, 3278, 3279, 3281, 3281, 3283, 3283, 3283, 3285, 3285, 3285, 3285, 3286, 3286, 3286, 3286, 3288, 3288, 3288, 3289, 3292, 3292, 3294, 3294, 3296, 3296, 3299, 3303, 3308, 3309, 3310, 3310, 3311, 3311, 3311, 3311, 3312, 3313, 3313, 3315, 3316, 3316, 3316, 3316, 3320, 3320, 3320, 3322, 3326, 3333, 3338, 3342, 3343, 3343, 3349, 3350, 3350, 3352, 3352, 3356, 3357, 3360, 3360, 3361, 3363, 3364, 3366, 3372, 3373, 3374, 3374, 3378, 3383, 3386, 3387, 3390, 3390, 3391, 3395, 3396, 3396, 3397, 3402, 3402, 3403, 3403, 3404, 3407, 3410, 3410, 3411, 3414, 3415, 3415, 3415, 3420, 3421, 3421, 3425, 3425, 3429, 3431, 3431, 3432, 3432, 3435, 3437, 3440, 3442, 3442, 3443, 3444, 3445, 3445, 3446, 3446, 3446, 3448, 3448, 3449, 3454, 3454, 3455, 3459, 3460, 3460, 3461, 3463, 3465, 3465, 3465, 3468, 3473, 3474, 3475, 3475, 3480, 3480, 3481, 3481, 3482, 3482, 3483, 3484, 3487, 3488, 3489, 3489, 3491, 3491, 3491, 3491, 3491, 3492, 3494, 3500, 3502, 3502, 3510, 3511, 3512, 3513, 3516, 3520, 3521, 3527, 3528, 3530, 3530, 3532, 3534, 3536, 3536, 3536, 3537, 3539, 3539, 3540, 3540, 3542, 3543, 3545, 3546, 3547, 3548, 3549, 3551, 3553, 3556, 3557, 3559, 3561, 3563, 3564, 3567, 3567, 3567, 3568, 3569, 3570, 3571, 3572, 3572, 3572, 3576, 3576, 3578, 3580, 3580, 3581, 3581, 3581, 3582, 3583, 3583, 3585, 3585, 3585, 3586, 3586, 3586, 3588, 3588, 3589, 3591, 3591, 3591, 3591, 3592, 3594, 3596, 3596, 3598, 3600, 3602, 3604, 3611, 3612, 3612, 3614, 3615, 3615, 3616, 3616, 3617, 3620, 3623, 3624, 3624, 3625, 3625, 3627, 3627, 3636, 3636, 3637, 3637, 3642, 3645, 3647, 3647, 3650, 3650, 3652, 3652, 3652, 3653, 3653, 3654, 3654, 3655, 3657, 3657, 3657, 3658, 3661, 3661, 3662, 3662, 3664, 3665, 3668, 3669, 3675, 3675, 3681, 3683, 3686, 3688, 3688, 3692, 3693, 3695, 3697, 3700, 3702, 3704, 3707, 3709, 3709, 3709, 3710, 3710, 3713, 3713, 3714, 3714, 3720, 3725, 3731, 3732, 3733, 3735, 3738, 3739, 3743, 3744, 3749, 3751, 3751, 3752, 3754, 3758, 3760, 3760, 3768, 3769, 3776, 3776, 3776, 3781, 3785, 3789, 3789, 3790, 3791, 3791, 3793, 3799, 3805, 3806, 3808, 3811, 3812, 3812, 3813, 3819, 3821, 3822, 3823, 3824, 3830, 3830, 3830, 3831, 3837, 3837, 3839, 3842, 3850, 3851, 3853, 3857, 3858, 3863, 3864, 3865, 3865, 3865, 3865, 3865, 3865, 3867, 3867, 3868, 3869, 3870, 3871, 3874, 3874, 3874, 3877, 3879, 3880, 3880, 3883, 3883, 3883, 3883, 3883, 3883, 3883, 3887, 3889, 3889, 3889, 3890, 3892, 3894, 3894, 3895, 3896, 3897, 3899, 3899, 3899, 3900, 3900, 3900, 3901, 3904, 3904, 3904, 3904, 3904, 3907, 3909, 3909, 3909, 3910, 3911, 3911, 3912, 3913, 3917, 3921, 3926, 3926, 3927, 3927, 3928, 3930, 3930, 3932, 3933, 3935, 3938, 3940, 3942, 3945, 3946, 3947, 3948, 3949, 3949, 3953, 3954, 3955, 3957, 3957, 3957, 3961, 3961, 3967, 3969, 3969, 3970, 3971, 3972, 3973, 3973, 3973, 3978, 3978, 3982, 3984, 3984, 3984, 3985, 3986, 3986, 3989, 3991, 3991, 3991, 3992, 3992, 3993, 4000, 4002, 4002, 4002, 4003, 4003, 4004, 4004, 4004, 4005, 4005, 4008, 4008, 4008, 4010, 4011, 4014, 4016, 4018, 4021, 4022, 4022, 4022, 4022, 4024, 4027, 4027, 4027, 4028, 4028, 4028, 4029, 4029, 4032, 4033, 4033, 4033, 4034, 4036, 4037, 4040, 4040, 4041, 4042, 4045, 4046, 4046, 4047, 4048, 4048, 4049, 4050, 4050, 4051, 4053, 4055, 4055, 4060, 4064, 4066, 4069, 4073, 4076, 4078, 4078, 4079, 4080, 4080, 4081, 4082, 4082, 4083, 4084, 4084, 4085, 4087, 4088, 4089, 4089, 4091, 4091, 4093, 4093, 4093, 4094, 4097, 4098, 4099, 4100, 4102, 4102, 4102, 4103, 4103, 4104, 4105, 4110, 4111, 4111, 4113, 4113, 4113, 4115, 4117, 4117, 4119, 4119, 4120, 4123, 4125, 4126, 4126, 4129, 4129, 4131, 4131, 4133, 4133, 4135, 4135, 4135, 4136, 4136, 4137, 4139, 4139, 4142, 4142, 4142, 4144, 4144, 4144, 4145, 4146, 4147, 4147, 4148, 4148, 4149, 4149, 4149, 4150, 4150, 4150, 4151, 4154, 4160, 4162, 4163, 4163, 4166, 4168, 4170, 4170, 4171, 4173, 4173, 4174, 4180, 4181, 4182, 4183, 4183, 4184, 4185, 4187, 4187, 4187, 4189, 4194, 4194, 4199, 4199, 4199, 4200, 4200, 4203, 4203, 4204, 4210, 4213, 4214, 4214, 4215, 4215, 4215, 4217, 4218, 4221, 4224, 4225, 4227, 4229, 4229, 4229, 4230, 4232, 4237, 4237, 4240, 4240, 4241, 4241, 4245, 4247, 4249, 4249, 4254, 4256, 4256, 4257, 4258, 4258, 4259, 4259, 4264, 4266, 4268, 4268, 4269, 4274, 4277, 4280, 4282, 4287, 4291, 4291, 4291, 4291, 4291, 4291, 4293, 4295, 4297, 4301, 4301, 4303, 4304, 4304, 4308, 4309, 4309, 4310, 4311, 4320, 4321, 4329, 4329, 4334, 4336, 4337, 4340, 4343, 4345, 4355, 4358, 4359, 4359, 4363, 4364, 4368, 4368, 4369, 4369, 4370, 4373, 4374, 4376, 4377, 4381, 4381, 4381, 4382, 4383, 4384, 4385, 4386, 4386, 4386, 4387, 4389, 4390, 4390, 4390, 4390, 4390, 4391, 4391, 4391, 4400, 4404, 4404, 4406, 4406, 4408, 4413, 4413, 4414, 4415, 4416, 4419, 4420, 4421, 4422, 4422, 4423, 4427, 4430, 4430, 4430, 4433, 4433, 4437, 4438, 4438, 4438, 4438, 4439, 4439, 4441, 4441, 4442, 4443, 4444, 4448, 4449, 4449, 4453, 4454, 4454, 4454, 4456, 4458, 4461, 4465, 4470, 4470, 4471, 4472, 4474, 4474, 4475, 4475, 4476, 4476, 4478, 4479, 4485, 4485, 4487, 4490, 4492, 4492, 4494, 4494, 4494, 4495, 4497, 4501, 4501, 4502, 4502, 4506, 4506, 4508, 4509, 4510, 4511, 4513, 4513, 4515, 4516, 4519, 4520, 4522, 4526, 4526, 4528, 4529, 4529, 4530, 4530, 4532, 4533, 4533, 4533, 4534, 4536, 4542, 4543, 4544, 4544, 4544, 4545, 4545, 4546, 4546, 4546, 4546, 4546, 4546, 4553, 4554, 4555, 4557, 4557, 4558, 4558, 4559, 4561, 4561, 4561, 4562, 4564, 4564, 4564, 4565, 4565, 4566, 4566, 4568, 4573, 4574, 4574, 4574, 4581, 4582, 4582, 4585, 4587, 4588, 4590, 4591, 4592, 4594, 4595, 4595, 4595, 4596, 4598, 4598, 4599, 4599, 4599, 4600, 4601, 4603, 4603, 4604, 4605, 4606, 4607, 4607, 4607, 4609, 4611, 4611, 4613, 4613, 4614, 4616, 4616, 4616, 4617, 4617, 4619, 4619, 4620, 4621, 4622, 4626, 4627, 4627, 4627, 4629, 4629, 4631, 4635, 4638, 4640, 4641, 4642, 4643, 4645, 4645, 4646, 4647, 4647, 4648, 4651, 4651, 4655, 4656, 4656, 4657, 4662, 4662, 4664, 4667, 4671, 4671, 4672, 4672, 4674, 4674, 4674, 4675, 4676, 4677, 4678, 4681, 4682, 4685, 4686, 4688, 4691, 4691, 4693, 4693, 4695, 4696, 4696, 4696, 4696, 4697, 4697, 4699, 4703, 4706, 4707, 4709, 4709, 4709, 4712, 4713, 4713, 4715, 4715, 4716, 4716, 4716, 4718, 4719, 4721, 4721, 4721, 4721, 4722, 4722, 4723, 4724, 4725, 4726, 4726, 4727, 4732, 4734, 4735, 4741, 4741, 4744, 4744, 4744, 4752, 4752, 4757, 4757, 4760, 4760, 4761, 4762, 4764, 4764, 4765, 4765, 4768, 4768, 4770, 4771, 4775, 4776, 4776, 4778, 4778, 4779, 4781, 4781, 4782, 4783, 4783, 4783, 4787, 4795, 4799, 4806, 4806, 4808, 4809, 4811, 4811, 4812, 4813, 4814, 4814, 4818, 4821, 4824, 4825, 4826, 4828, 4831, 4831, 4831, 4831, 4835, 4835, 4836, 4836, 4836, 4837, 4838, 4842, 4846, 4851, 4853, 4856, 4856, 4856, 4857, 4857, 4858, 4860, 4860, 4860, 4860, 4860, 4860, 4860, 4861, 4862, 4863, 4863, 4867, 4868, 4868, 4868, 4869, 4869, 4871, 4873, 4873, 4874, 4875, 4875, 4875, 4875, 4880, 4881, 4883, 4883, 4884, 4890, 4890, 4891, 4893, 4900, 4901, 4901, 4906, 4906, 4908, 4909, 4910, 4917, 4919, 4921, 4924, 4926, 4927, 4928, 4929, 4931, 4933, 4933, 4935, 4936, 4937, 4938, 4939, 4940, 4940, 4940, 4941, 4942, 4942, 4945, 4945, 4945, 4947, 4948, 4949, 4949, 4949, 4953, 4956, 4960, 4963, 4963, 4964, 4964, 4968, 4969, 4971, 4971, 4971, 4971, 4973, 4974, 4975, 4982, 4984, 4984, 4984, 4986, 4986, 4986, 4987, 4987, 4988, 4990, 4997, 5003, 5003, 5009, 5010, 5014, 5017, 5018, 5021, 5022, 5025, 5026, 5027, 5028, 5034, 5035, 5036, 5037, 5037, 5038, 5039, 5043, 5044, 5047, 5050, 5050, 5051, 5056, 5058, 5060, 5061, 5063, 5064, 5066, 5069, 5072, 5074, 5075, 5076, 5076, 5078, 5079, 5081, 5083, 5083, 5083, 5092, 5095, 5095, 5097, 5100, 5104, 5104, 5106, 5106, 5107, 5108, 5111, 5111, 5115, 5116, 5119, 5120, 5123, 5123, 5124, 5125, 5125, 5125, 5128, 5130, 5133, 5133, 5134, 5134, 5135, 5136, 5139, 5140, 5142, 5144, 5146, 5146, 5146, 5146, 5147, 5147, 5149, 5149, 5150, 5150, 5151, 5151, 5153, 5153, 5160, 5161, 5163, 5165, 5165, 5165, 5165, 5165, 5166, 5175, 5176, 5178, 5182, 5184, 5185, 5185, 5186, 5186, 5189, 5190, 5191, 5192, 5192, 5192, 5192, 5194, 5194, 5195, 5196, 5198, 5199, 5199, 5200, 5202, 5203, 5211, 5211, 5211, 5213, 5214, 5215, 5217, 5218, 5218, 5219, 5220, 5221, 5221, 5225, 5225, 5225, 5225, 5226, 5229, 5230, 5230, 5231, 5234, 5234, 5234, 5238, 5239, 5241, 5243, 5243, 5244, 5244, 5244, 5245, 5245, 5251, 5255, 5255, 5256, 5257, 5257, 5257, 5265, 5265, 5265, 5265, 5265, 5265, 5268, 5269, 5269, 5273, 5273, 5274, 5274, 5274, 5275, 5275, 5275, 5276, 5276, 5276, 5278, 5279, 5279, 5281, 5281, 5282, 5282, 5286, 5287, 5287, 5288, 5288, 5290, 5297, 5299, 5302, 5302, 5303, 5303, 5304, 5306, 5308, 5308, 5309, 5309, 5316, 5321, 5324, 5324, 5325, 5325, 5326, 5326, 5328, 5331, 5332, 5332, 5335, 5335, 5337, 5337, 5341, 5341, 5341, 5341, 5342, 5343, 5343, 5343, 5344, 5345, 5345, 5348, 5348, 5352, 5352, 5357, 5359, 5359, 5364, 5365, 5366, 5366, 5367, 5367, 5367, 5369, 5372, 5372, 5372, 5377, 5378, 5378, 5382, 5383, 5391, 5394, 5394, 5397, 5397, 5399, 5406, 5406, 5408, 5408, 5409, 5409, 5411, 5413, 5413, 5414, 5416, 5420, 5420, 5421, 5424, 5426, 5426, 5429, 5430, 5434, 5434, 5434, 5437, 5437, 5438, 5440, 5441, 5441, 5442, 5443, 5444, 5444, 5446, 5449, 5449, 5449, 5451, 5453, 5456, 5462, 5463, 5463, 5467, 5474, 5480, 5484, 5485, 5485, 5488, 5488, 5489, 5489, 5490, 5490, 5492, 5492, 5492, 5496, 5496, 5499, 5505, 5507, 5507, 5507, 5508, 5509, 5509, 5512, 5514, 5514, 5516, 5517, 5517, 5518, 5519, 5521, 5521, 5525, 5526, 5527, 5528, 5530, 5533, 5535, 5535, 5538, 5538, 5541, 5543, 5546, 5547, 5551, 5551, 5554, 5556, 5557, 5558, 5558, 5558, 5562, 5563, 5564, 5567, 5567, 5571, 5572, 5572, 5573, 5574, 5574, 5575, 5578, 5578, 5579, 5579, 5579, 5581, 5581, 5585, 5586, 5588, 5590, 5590, 5595, 5595, 5602, 5602, 5602, 5606, 5607, 5607, 5607, 5609, 5610, 5610, 5611, 5611, 5611, 5613, 5613, 5616, 5617, 5618, 5620, 5623, 5623, 5625, 5626, 5630, 5634, 5634, 5636, 5646, 5646, 5646, 5648, 5652, 5652, 5653, 5656, 5658, 5658, 5661, 5665, 5676, 5676, 5676, 5676, 5680, 5680, 5687, 5687, 5688, 5692, 5693, 5693, 5702, 5707, 5714, 5714, 5714, 5721, 5722, 5723, 5728, 5728, 5737, 5737, 5740, 5746, 5746, 5747, 5752, 5758, 5760, 5762, 5762, 5764, 5764, 5766, 5766, 5767, 5772, 5773, 5774, 5774, 5775, 5778, 5780, 5781, 5783, 5783, 5783, 5783, 5784, 5785, 5786, 5786, 5786, 5794, 5801, 5801, 5802, 5803, 5806, 5808, 5809, 5813, 5813, 5819, 5822, 5824, 5830, 5831, 5833, 5833, 5837, 5843, 5846, 5847, 5848, 5851, 5851, 5851, 5851, 5852, 5852, 5854, 5859, 5861, 5863, 5864, 5867, 5870, 5871, 5872, 5872, 5879, 5881, 5881, 5883, 5884, 5886, 5888, 5890, 5890, 5890, 5891, 5891, 5894, 5899, 5899, 5900, 5905, 5906, 5907, 5907, 5908, 5908, 5910, 5910, 5910, 5911, 5912, 5912, 5913, 5915, 5918, 5919, 5920, 5925, 5928, 5930, 5930, 5930, 5932, 5934, 5934, 5934, 5934, 5935, 5942, 5948, 5949, 5951, 5952, 5953, 5954, 5955, 5955, 5957, 5957, 5957, 5959, 5961, 5961, 5961, 5969, 5969, 5970, 5973, 5974, 5975, 5975, 5975, 5977, 5980, 5980, 5982, 5983, 5984, 5986, 5986, 5986, 5986, 5988, 5988, 5988, 5990, 5992, 5992, 5992, 5994, 5995, 5998, 6001, 6003, 6004, 6004, 6004, 6004, 6004, 6004, 6009, 6009, 6010, 6014, 6016, 6019, 6019, 6023, 6023, 6023, 6024, 6028, 6029, 6030, 6031, 6031, 6033, 6033, 6033, 6033, 6033, 6034, 6036, 6038, 6040, 6041, 6041, 6047, 6048, 6049, 6049, 6050, 6050, 6052, 6052, 6053, 6055, 6057, 6064, 6065, 6066, 6068, 6068, 6070, 6073, 6074, 6074, 6075, 6076, 6078, 6078, 6081, 6082, 6082, 6083, 6084, 6084, 6085, 6086, 6087, 6087, 6088, 6088, 6088, 6088, 6089, 6090, 6092, 6092, 6093, 6094, 6095, 6095, 6095, 6096, 6097, 6097, 6098, 6098, 6099, 6100, 6101, 6101, 6104, 6104, 6106, 6106, 6106, 6107, 6107, 6107, 6107, 6108, 6108, 6111, 6111, 6112, 6112, 6112, 6113, 6114, 6114, 6115, 6115, 6116, 6116, 6116, 6117, 6118, 6119, 6120, 6121, 6125, 6126, 6127, 6127, 6127, 6127, 6128, 6129, 6129, 6130, 6131, 6132, 6134, 6134, 6135, 6135, 6135, 6136, 6136, 6136, 6136, 6137, 6138, 6140, 6141, 6143, 6143, 6143, 6143, 6143, 6144, 6144, 6145, 6145, 6145, 6147, 6147, 6147, 6148, 6149, 6150, 6151, 6151, 6151, 6151, 6152, 6152, 6154, 6154, 6156, 6156, 6156, 6156, 6157, 6158, 6160, 6160, 6161, 6161, 6161, 6161, 6161, 6163, 6164, 6164, 6164, 6166, 6167, 6167, 6167, 6168, 6169, 6170, 6170, 6172, 6174, 6174, 6174, 6179, 6180, 6180, 6182, 6183, 6183, 6183, 6184, 6184, 6185, 6185, 6186, 6186, 6187, 6187, 6190, 6191, 6192, 6196, 6197, 6198, 6198, 6198, 6199, 6200, 6201, 6201, 6202, 6203, 6203, 6203, 6203, 6208, 6210, 6212, 6212, 6212, 6214, 6214, 6217, 6220, 6220, 6223, 6226, 6226, 6227, 6227, 6228, 6228, 6228, 6233, 6234, 6235, 6235, 6237, 6238, 6238, 6244, 6244, 6244, 6245, 6247, 6252, 6253, 6253, 6255, 6255, 6257, 6257, 6259, 6259, 6259, 6261, 6266, 6268, 6268, 6269, 6270, 6270, 6271, 6271, 6271, 6271, 6274, 6275, 6275, 6276, 6276, 6277, 6279, 6280, 6281, 6281, 6282, 6284, 6287, 6287, 6288, 6289, 6290, 6291, 6292, 6294, 6294, 6294, 6295, 6295, 6296, 6296, 6296, 6297, 6299, 6299, 6299, 6299, 6300, 6301, 6301, 6302, 6302, 6303, 6303, 6304, 6304, 6304, 6304, 6304, 6305, 6305, 6306, 6308, 6309, 6310, 6310, 6314, 6314, 6314, 6315, 6316, 6318, 6318, 6319, 6319, 6319, 6320, 6320, 6320, 6321, 6322, 6324, 6324, 6324, 6325, 6325, 6326, 6327, 6327, 6327, 6328, 6329, 6330, 6331, 6332, 6333, 6334, 6335, 6336, 6336, 6336, 6337, 6338, 6338, 6338, 6339, 6339, 6340, 6341, 6341, 6343, 6343, 6344, 6345, 6346, 6347, 6347, 6348, 6349, 6350, 6350, 6350, 6350, 6352, 6353, 6353, 6355, 6361, 6362, 6362, 6367, 6372, 6376, 6376, 6377, 6377, 6379, 6380, 6380, 6380, 6382, 6382, 6385, 6385, 6388, 6388, 6389, 6393, 6393, 6395, 6402, 6403, 6403, 6405, 6406, 6406, 6406, 6408, 6408, 6411, 6411, 6417, 6419, 6419, 6420, 6423, 6423, 6424, 6427, 6427, 6427, 6429, 6429, 6432, 6436, 6438, 6438, 6438, 6441, 6443, 6444, 6446, 6447, 6447, 6448, 6448, 6450, 6451, 6452, 6454, 6457, 6458, 6460, 6461, 6463, 6463, 6467, 6468, 6468, 6468, 6469, 6469, 6476, 6477, 6477, 6481, 6481, 6482, 6483, 6483, 6487, 6489, 6496, 6497, 6497, 6497, 6498, 6498, 6498, 6500, 6503, 6503, 6504, 6508, 6511, 6512, 6512, 6516, 6518, 6521, 6521, 6527, 6527, 6527, 6529, 6529, 6530, 6531, 6533, 6533, 6533, 6534, 6537, 6538, 6540, 6540, 6541, 6544, 6544, 6544, 6544, 6547, 6549, 6551, 6551, 6552, 6553, 6553, 6554, 6555, 6556, 6557, 6559, 6562, 6562, 6562, 6563, 6564, 6564, 6565, 6566, 6566, 6566, 6567, 6570, 6570, 6571, 6571, 6571, 6572, 6572, 6572, 6574, 6575, 6575, 6576, 6577, 6577, 6578, 6578, 6579, 6581, 6582, 6582, 6582, 6582, 6582, 6584, 6588, 6590, 6596, 6597, 6597, 6599, 6599, 6600, 6600, 6603, 6605, 6607, 6607, 6607, 6608, 6609, 6611, 6611, 6611, 6612, 6612, 6612, 6614, 6616, 6616, 6617, 6618, 6618, 6618, 6618, 6619, 6621, 6622, 6625, 6625, 6625, 6627, 6627, 6628, 6630, 6636, 6637, 6639, 6639, 6640, 6643, 6643, 6644, 6644, 6646, 6646, 6648, 6651, 6652, 6654, 6655, 6655, 6658, 6658, 6659, 6663, 6663, 6664, 6665, 6671, 6672, 6673, 6673, 6674, 6674, 6675, 6675, 6677, 6677, 6678, 6682, 6683, 6684, 6684, 6686, 6687, 6691, 6692, 6695, 6697, 6699, 6702, 6706, 6707, 6713, 6713, 6715, 6716, 6716, 6717, 6719, 6720, 6723, 6724, 6724, 6725, 6726, 6727, 6727, 6728, 6729, 6729, 6731, 6732, 6733, 6734, 6735, 6735, 6737, 6737, 6737, 6738, 6738, 6740, 6740, 6740, 6743, 6745, 6747, 6753, 6753, 6753, 6756, 6760, 6760, 6763, 6764, 6764, 6765, 6765, 6765, 6765, 6766, 6770, 6773, 6774, 6774, 6775, 6775, 6777, 6779, 6785, 6785, 6785, 6788, 6788, 6792, 6799, 6801, 6801, 6801, 6803, 6803, 6806, 6808, 6808, 6809, 6813, 6814, 6814, 6815, 6816, 6816, 6817, 6823, 6823, 6823, 6824, 6824, 6824, 6826, 6826, 6827, 6829, 6831, 6833, 6837, 6839, 6839, 6840, 6840, 6844, 6851, 6853, 6855, 6859, 6860, 6862, 6871, 6872, 6874, 6875, 6876, 6877, 6880, 6880, 6887, 6890, 6892, 6893, 6895, 6895, 6896, 6896, 6899, 6899, 6899, 6900, 6901, 6901, 6901, 6902, 6902, 6903, 6906, 6908, 6908, 6908, 6908, 6913, 6914, 6914, 6915, 6915, 6916, 6916, 6918, 6920, 6922, 6923, 6923, 6925, 6925, 6930, 6930, 6932, 6932, 6933, 6933, 6935, 6939, 6939, 6939, 6942, 6942, 6943, 6943, 6945, 6945, 6947, 6947, 6947, 6948, 6949, 6950, 6951, 6952, 6952, 6957, 6959, 6960, 6961, 6963, 6963, 6965, 6967, 6969, 6969, 6969, 6969, 6971, 6973, 6974, 6975, 6975, 6975, 6976, 6977, 6977, 6978, 6978, 6979, 6980, 6982, 6982, 6983, 6984, 6985, 6990, 6997, 6998, 7000, 7004, 7006, 7007, 7007, 7007, 7009, 7014, 7014, 7014, 7014, 7014, 7014, 7014, 7016, 7016, 7017, 7018, 7023, 7025, 7025, 7030, 7030, 7031, 7031, 7031, 7034, 7034, 7037, 7037, 7038, 7039, 7039, 7042, 7043, 7045, 7046, 7046, 7046, 7047, 7049, 7050, 7053, 7055, 7057, 7060, 7060, 7068, 7071, 7075, 7077, 7080, 7082, 7085, 7085, 7087, 7088, 7089, 7091, 7095, 7098, 7100, 7101, 7103, 7103, 7104, 7106, 7108, 7108, 7109, 7109, 7110, 7114, 7116, 7116, 7120, 7121, 7121, 7123, 7128, 7133, 7133, 7134, 7135, 7139, 7139, 7141, 7143, 7143, 7145, 7145, 7148, 7148, 7148, 7150, 7156, 7157, 7158, 7161, 7161, 7162, 7162, 7164, 7164, 7164, 7164, 7164, 7166, 7171, 7172, 7172, 7174, 7180, 7181, 7183, 7184, 7187, 7187, 7192, 7193, 7194, 7194, 7195, 7199, 7202, 7203, 7204, 7207, 7212, 7213, 7215, 7215, 7215, 7215, 7216, 7216, 7219, 7219, 7222, 7223, 7227, 7227, 7228, 7230, 7230, 7232, 7235, 7235, 7236, 7237, 7239, 7239, 7239, 7241, 7241, 7241, 7242, 7243, 7243, 7243, 7243, 7245, 7245, 7249, 7253, 7254, 7254, 7255, 7257, 7261, 7262, 7262, 7262, 7263, 7263, 7263, 7264, 7264, 7265, 7267, 7267, 7272, 7273, 7274, 7275, 7276, 7277, 7279, 7279, 7281, 7281, 7283, 7284, 7284, 7284, 7284, 7286, 7287, 7290, 7292, 7296, 7299, 7301, 7302, 7302, 7307, 7307, 7307, 7310, 7311, 7311, 7311, 7311, 7312, 7315, 7316, 7317, 7318, 7318, 7318, 7318, 7318, 7320, 7320, 7320, 7320, 7321, 7322, 7323, 7323, 7324, 7325, 7328, 7328, 7329, 7329, 7330, 7330, 7331, 7332, 7332, 7332, 7333, 7333, 7335, 7335, 7335, 7335, 7338, 7338, 7338, 7338, 7338, 7339, 7339, 7339, 7339, 7339, 7340, 7340, 7340, 7340, 7346, 7348, 7352, 7353, 7354, 7356, 7357, 7357, 7358, 7360, 7363, 7364, 7364, 7364, 7364, 7367, 7367, 7369, 7369, 7371, 7372, 7373, 7374, 7374, 7377, 7378, 7378, 7378, 7378, 7378, 7379, 7380, 7382, 7382, 7383, 7383, 7383, 7383, 7383, 7385, 7386, 7387, 7387, 7389, 7390, 7391, 7391, 7392, 7396, 7398, 7400, 7400, 7400, 7403, 7403, 7405, 7406, 7406, 7406, 7409, 7411, 7413, 7414, 7419, 7419, 7420, 7425, 7425, 7426, 7426, 7426, 7428, 7428, 7428, 7428, 7428, 7432, 7433, 7434, 7434, 7436, 7437, 7439, 7439, 7442, 7443, 7443, 7448, 7448, 7449, 7450, 7450, 7451, 7451, 7454, 7454, 7454, 7457, 7461, 7461, 7462, 7463, 7464, 7464, 7466, 7467, 7467, 7468, 7468, 7468, 7468, 7468, 7469, 7470, 7471, 7472, 7472, 7472, 7473, 7473, 7474, 7474, 7474, 7474, 7474, 7475, 7475, 7476, 7476, 7476, 7477, 7480, 7480, 7481, 7482, 7483, 7489, 7489, 7490, 7493, 7497, 7497, 7498, 7501, 7501, 7502, 7503, 7503, 7503, 7505, 7506, 7507, 7507, 7507, 7508, 7508, 7510, 7511, 7511, 7511, 7511, 7512, 7514, 7514, 7514, 7514, 7517, 7518, 7519, 7521, 7521, 7524, 7525, 7525, 7535, 7535, 7536, 7536, 7536, 7539, 7539, 7540, 7542, 7543, 7543, 7545, 7545, 7546, 7546, 7548, 7548, 7549, 7556, 7562, 7564, 7566, 7566, 7569, 7569, 7569, 7570, 7571, 7571, 7571, 7576, 7578, 7578, 7579, 7580, 7581, 7582, 7582, 7583, 7590, 7591, 7592, 7598, 7599, 7602, 7609, 7611, 7611, 7613, 7615, 7615, 7616, 7618, 7619, 7620, 7621, 7621, 7622, 7622, 7623, 7623, 7624, 7625, 7626, 7627, 7627, 7627, 7627, 7632, 7632, 7635, 7636, 7637, 7637, 7638, 7641, 7642, 7642, 7642, 7642, 7643, 7643, 7644, 7645, 7645, 7645, 7646, 7646, 7648, 7652, 7652, 7654, 7655, 7655, 7655, 7655, 7655, 7657, 7657, 7657, 7661, 7661, 7661, 7662, 7664, 7665, 7666, 7667, 7668, 7668, 7668, 7670, 7671, 7674, 7674, 7675, 7676, 7676, 7677, 7678, 7678, 7678, 7678, 7679, 7680, 7680, 7681, 7681, 7681, 7682, 7682, 7682, 7683, 7687, 7688, 7688, 7688, 7689, 7689, 7690, 7690, 7692, 7694, 7695, 7696, 7697, 7699, 7699, 7700, 7702, 7703, 7709, 7712, 7713, 7715, 7715, 7718, 7721, 7725, 7726, 7733, 7734, 7734, 7736, 7736, 7737, 7740, 7742, 7746, 7747, 7750, 7750, 7751, 7756, 7758, 7758, 7758, 7758, 7758, 7762, 7763, 7765, 7767, 7767, 7770, 7770, 7771, 7773, 7773, 7774, 7775, 7775, 7776, 7783, 7785, 7787, 7792, 7793, 7795, 7800, 7801, 7804, 7805, 7817, 7817, 7820, 7822, 7822, 7822, 7825, 7831, 7833, 7834, 7834, 7837, 7837, 7850, 7851, 7852, 7858, 7861, 7865, 7865, 7868, 7870, 7870, 7870, 7872, 7874, 7874, 7875, 7875, 7875, 7876, 7876, 7876, 7876, 7878, 7880, 7883, 7889, 7892, 7893, 7894, 7895, 7895, 7896, 7896, 7902, 7904, 7905, 7906, 7917, 7917, 7919, 7919, 7919, 7920, 7920, 7923, 7929, 7929, 7929, 7929, 7929, 7929, 7935, 7940, 7940, 7947, 7949, 7952, 7956, 7958, 7958, 7959, 7960, 7960, 7960, 7961, 7968, 7972, 7974, 7974, 7974, 7978, 7979, 7979, 7982, 7983, 7983, 7986, 7986, 7986, 7986, 7987, 7988, 7989, 7990, 7990, 7991, 7991, 7992, 7994, 7994, 7994, 7996, 7996, 7998, 7999, 8000, 8000, 8000, 8001, 8002, 8005, 8005, 8006, 8007, 8008, 8008, 8009, 8010, 8010, 8011, 8012, 8012, 8013, 8016, 8016, 8016, 8017, 8017, 8017, 8020, 8021, 8023, 8023, 8024, 8025, 8025, 8026, 8027, 8027, 8030, 8031, 8031, 8031, 8032, 8035, 8036, 8036, 8037, 8043, 8043, 8044, 8044, 8047, 8047, 8050, 8051, 8051, 8051, 8054, 8058, 8058, 8059, 8059, 8064, 8066, 8068, 8072, 8075, 8078, 8079, 8080, 8081, 8081, 8084, 8084, 8087, 8088, 8096, 8096, 8097, 8098, 8099, 8100, 8103, 8105, 8106, 8106, 8107, 8111, 8111, 8116, 8117, 8119, 8122, 8123, 8123, 8126, 8127, 8128, 8132, 8132, 8133, 8133, 8135, 8143, 8143, 8143, 8145, 8145, 8146, 8146, 8146, 8147, 8148, 8148, 8148, 8150, 8153, 8155, 8156, 8156, 8156, 8156, 8157, 8157, 8158, 8160, 8162, 8164, 8164, 8164, 8165, 8166, 8166, 8166, 8167, 8167, 8168, 8168, 8169, 8169, 8169, 8170, 8170, 8172, 8174, 8177, 8178, 8178, 8178, 8179, 8179, 8180, 8181, 8181, 8181, 8181, 8183, 8183, 8183, 8183, 8186, 8186, 8187, 8187, 8189, 8190, 8190, 8191, 8195, 8195, 8198, 8198, 8200, 8200, 8200, 8202, 8203, 8203, 8203, 8203, 8204, 8204, 8204, 8204, 8205, 8205, 8205, 8206, 8206, 8207, 8210, 8212, 8213, 8215, 8215, 8216, 8217, 8217, 8217, 8217, 8218, 8219, 8219, 8221, 8221, 8222, 8222, 8222, 8222, 8222, 8222, 8223, 8226, 8227, 8230, 8231, 8231, 8231, 8236, 8236, 8238, 8238, 8238, 8241, 8241, 8242, 8246, 8246, 8248, 8248, 8248, 8249, 8258, 8258, 8262, 8269, 8269, 8269, 8272, 8272, 8272, 8272, 8274, 8275, 8275, 8282, 8283, 8284, 8284, 8287, 8287, 8287, 8289, 8289, 8292, 8292, 8295, 8298, 8300, 8300, 8306, 8306, 8307, 8308, 8311, 8316, 8318, 8319, 8322, 8322, 8327, 8328, 8330, 8333, 8336, 8339, 8339, 8341, 8342, 8344, 8348, 8348, 8348, 8348, 8350, 8350, 8350, 8351, 8354, 8359, 8361, 8368, 8368, 8368, 8369, 8373, 8375, 8377, 8377, 8379, 8380, 8382, 8383, 8385, 8386, 8388, 8388, 8388, 8389, 8390, 8392, 8394, 8394, 8394, 8396, 8396, 8398, 8404, 8404, 8404, 8404, 8405, 8405, 8407, 8408, 8408, 8412, 8412, 8413, 8414, 8415, 8416, 8416, 8416, 8418, 8418, 8420, 8421, 8421, 8421, 8421, 8421, 8421, 8421, 8422, 8422, 8424, 8426, 8426, 8431, 8437, 8437, 8438, 8438, 8438, 8440, 8440, 8441, 8441, 8447, 8447, 8448, 8449, 8450, 8455, 8456, 8456, 8457, 8457, 8457, 8459, 8459, 8459, 8460, 8462, 8462, 8464, 8465, 8466, 8467, 8468, 8468, 8469, 8471, 8472, 8472, 8475, 8475, 8477, 8483, 8483, 8483, 8483, 8486, 8486, 8487, 8487, 8492, 8493, 8498, 8498, 8503, 8505, 8505, 8507, 8509, 8511, 8511, 8511, 8511, 8511, 8512, 8513, 8513, 8513, 8513, 8513, 8514, 8514, 8515, 8515, 8515, 8517, 8517, 8517, 8519, 8520, 8520, 8520, 8520, 8520, 8521, 8521, 8521, 8523, 8524, 8524, 8524, 8524, 8525, 8529, 8530, 8531, 8531, 8531, 8532, 8532, 8532, 8532, 8534, 8534, 8535, 8535, 8536, 8537, 8538, 8538, 8538, 8538, 8538, 8538, 8538, 8538, 8538, 8539, 8539, 8539, 8539, 8539, 8541, 8541, 8541, 8541, 8541, 8541, 8541, 8542, 8543, 8544, 8544, 8548, 8548, 8548, 8549, 8549, 8549, 8549, 8551, 8551, 8552, 8552, 8552, 8552, 8552, 8554, 8554, 8554, 8555, 8555, 8555, 8556, 8556, 8557, 8557, 8557, 8558, 8558, 8558, 8559, 8559, 8559, 8562, 8562, 8565, 8566, 8569, 8570, 8570, 8571, 8573, 8574, 8577, 8579, 8582, 8585, 8586, 8586, 8586, 8588, 8588, 8588, 8588, 8588, 8588, 8588, 8589, 8590, 8592, 8594, 8594, 8595, 8595, 8597, 8600, 8602, 8604, 8604, 8605, 8605, 8605, 8606, 8607, 8610, 8610, 8613, 8613, 8614, 8615, 8617, 8618, 8619, 8619, 8620, 8621, 8623, 8623, 8624, 8625, 8627, 8628, 8628, 8628, 8628, 8629, 8629, 8629, 8631, 8632, 8633, 8633, 8633, 8634, 8634, 8635, 8635, 8635, 8636, 8636, 8638, 8638, 8639, 8640, 8641, 8642, 8645, 8645, 8645, 8646, 8647, 8649, 8652, 8654, 8661, 8664, 8666, 8671, 8673, 8674, 8675, 8675, 8677, 8677, 8677, 8679, 8680, 8683, 8683, 8683, 8686, 8686, 8687, 8688, 8688, 8689, 8689, 8690, 8690, 8691, 8691, 8692, 8702, 8704, 8705, 8705, 8706, 8706, 8706, 8707, 8707, 8707, 8709, 8709, 8709, 8709, 8709, 8709, 8709, 8710, 8714, 8715, 8715, 8715, 8718, 8718, 8719, 8719, 8720, 8722, 8723, 8723, 8723, 8724, 8726, 8729, 8734, 8735, 8737, 8737, 8737, 8739, 8739, 8740, 8740, 8740, 8741, 8741, 8742, 8744, 8744, 8744, 8745, 8745, 8748, 8751, 8752, 8752, 8761, 8761, 8762, 8762, 8763, 8764, 8764, 8765, 8767, 8768, 8768, 8769, 8769, 8770, 8771, 8771, 8772, 8772, 8772, 8772, 8773, 8774, 8779, 8779, 8779, 8780, 8784, 8790, 8790, 8797, 8797, 8798, 8799, 8802, 8803, 8806, 8806, 8806, 8806, 8806, 8806, 8808, 8809, 8810, 8811, 8813, 8815, 8816, 8816, 8816, 8816, 8819, 8819, 8820, 8821, 8821, 8821, 8821, 8822, 8822, 8822, 8824, 8824, 8824, 8825, 8826, 8826, 8828, 8828, 8828, 8834, 8837, 8838, 8838, 8839, 8845, 8846, 8851, 8851, 8852, 8852, 8855, 8855, 8855, 8856, 8857, 8859, 8861, 8864, 8865, 8865, 8865, 8867, 8873, 8874, 8874, 8884, 8890, 8892, 8892, 8892, 8897, 8899, 8899, 8909, 8910, 8910, 8915, 8915, 8915, 8915, 8916, 8919, 8920, 8922, 8922, 8924, 8924, 8925, 8925, 8926, 8926, 8926, 8926, 8932, 8932, 8933, 8933, 8934, 8937, 8937, 8938, 8938, 8939, 8941, 8941, 8941, 8941, 8942, 8944, 8944, 8947, 8949, 8952, 8952, 8952, 8953, 8955, 8957, 8957, 8957, 8957, 8957, 8957, 8957, 8957, 8958, 8958, 8958, 8958, 8959, 8959, 8959, 8960, 8963, 8963, 8963, 8964, 8967, 8967, 8968, 8969, 8972, 8974, 8976, 8977, 8977, 8979, 8979, 8980, 8982, 8984, 8984, 8985, 8986, 8987, 8987, 8994, 8996, 8996, 8997, 8998, 8998, 8998, 9000, 9000, 9000, 9000, 9001, 9002, 9003, 9004, 9007, 9007, 9011, 9013, 9013, 9017, 9017, 9017, 9018, 9020, 9023, 9026, 9027, 9028, 9029, 9029, 9034, 9034, 9035, 9035, 9036, 9037, 9038, 9038, 9038, 9038, 9039, 9042, 9042, 9043, 9044, 9044, 9047, 9050, 9050, 9050, 9053, 9054, 9054, 9054, 9058, 9058, 9063, 9064, 9066, 9067, 9069, 9069, 9070, 9071, 9076, 9076, 9077, 9077, 9080, 9083, 9085, 9085, 9087, 9087, 9089, 9092, 9092, 9094, 9094, 9095, 9095, 9095, 9096, 9100, 9101, 9101, 9102, 9102, 9103, 9103, 9103, 9103, 9103, 9104, 9104, 9105, 9107, 9114, 9114, 9114, 9114, 9115, 9116, 9118, 9118, 9119, 9119, 9121, 9122, 9122, 9124, 9125, 9125, 9125, 9126, 9127, 9128, 9128, 9129, 9129, 9129, 9129, 9129, 9129, 9130, 9132, 9135, 9135, 9136, 9138, 9140, 9141, 9141, 9142, 9142, 9142, 9143, 9143, 9143, 9144, 9145, 9146, 9150, 9151, 9151, 9151, 9154, 9154, 9155, 9155, 9156, 9157, 9157, 9158, 9158, 9158, 9159, 9159, 9161, 9161, 9161, 9164, 9165, 9168, 9170, 9172, 9172, 9173, 9173, 9175, 9177, 9178, 9181, 9181, 9182, 9183, 9184, 9184, 9185, 9185, 9188, 9188, 9188, 9188, 9188, 9191, 9191, 9194, 9194, 9195, 9195, 9197, 9200, 9200, 9201, 9203, 9207, 9209, 9210, 9213, 9219, 9221, 9221, 9221, 9223, 9224, 9224, 9224, 9224, 9224, 9225, 9226, 9226, 9227, 9228, 9229, 9230, 9230, 9231, 9235, 9235, 9237, 9240, 9240, 9240, 9242, 9242, 9242, 9242, 9242, 9244, 9246, 9248, 9248, 9249, 9251, 9251, 9251, 9256, 9256, 9260, 9264, 9266, 9266, 9267, 9268, 9268, 9268, 9270, 9270, 9270, 9270, 9270, 9271, 9277, 9277, 9278, 9280, 9282, 9286, 9286, 9286, 9288, 9289, 9289, 9290, 9291, 9299, 9299, 9309, 9311, 9312, 9314, 9318, 9321, 9323, 9323, 9324, 9325, 9327, 9330, 9337, 9337, 9345, 9345, 9346, 9347, 9348, 9353, 9355, 9365, 9367, 9367, 9370, 9371, 9373, 9375, 9375, 9376, 9378, 9379, 9379, 9380, 9382, 9382, 9383, 9386, 9387, 9387, 9389, 9391, 9393, 9393, 9393, 9396, 9396, 9396, 9397, 9399, 9400, 9400, 9401, 9403, 9403, 9403, 9405, 9405, 9407, 9407, 9408, 9411, 9413, 9415, 9418, 9420, 9422, 9423, 9432, 9433, 9433, 9433, 9433, 9435, 9435, 9436, 9437, 9438, 9443, 9446, 9451, 9456, 9457, 9460, 9460, 9461, 9463, 9469, 9470, 9471, 9471, 9472, 9472, 9479, 9480, 9480, 9480, 9481, 9481, 9481, 9484, 9485, 9489, 9490, 9490, 9493, 9498, 9499, 9499, 9504, 9505, 9508, 9509, 9513, 9513, 9514, 9516, 9517, 9518, 9519, 9521, 9522, 9523, 9526, 9527, 9529, 9529, 9530, 9534, 9534, 9534, 9534, 9535, 9535, 9536, 9536, 9536, 9539, 9539, 9539, 9540, 9542, 9544, 9545, 9545, 9546, 9546, 9548, 9549, 9549, 9551, 9552, 9553, 9556, 9556, 9556, 9557, 9557, 9557, 9561, 9561, 9562, 9563, 9565, 9565, 9565, 9566, 9568, 9569, 9570, 9571, 9573, 9574, 9574, 9574, 9575, 9576, 9576, 9579, 9581, 9581, 9582, 9582, 9585, 9587, 9588, 9588, 9589, 9594, 9594, 9596, 9596, 9596, 9596, 9596, 9598, 9598, 9600, 9604, 9607, 9607, 9608, 9609, 9610, 9610, 9611, 9612, 9613, 9618, 9618, 9618, 9619, 9620, 9620, 9621, 9624, 9624, 9624, 9624, 9625, 9625, 9625, 9625, 9625, 9625, 9625, 9627, 9627, 9630, 9630, 9630, 9632, 9638, 9639, 9640, 9641, 9642, 9643, 9643, 9646, 9647, 9647, 9649, 9652, 9654, 9655, 9657, 9658, 9660, 9662, 9664, 9665, 9665, 9668, 9670, 9673, 9673, 9674, 9677, 9678, 9679, 9679, 9682, 9684, 9684, 9693, 9693, 9694, 9696, 9696, 9699, 9700, 9701, 9704, 9707, 9707, 9714, 9714, 9718, 9720, 9722, 9723, 9723, 9724, 9727, 9728, 9729, 9730, 9730, 9731, 9733, 9733, 9733, 9734, 9734, 9734, 9738, 9738, 9738, 9739, 9740, 9740, 9742, 9743, 9744, 9744, 9744, 9745, 9745, 9747, 9747, 9749, 9753, 9753, 9753, 9753, 9754, 9756, 9758, 9758, 9760, 9760, 9764, 9764, 9765, 9768, 9768, 9770, 9771, 9772, 9772, 9774, 9774, 9775, 9777, 9778, 9778, 9778, 9779, 9780, 9784, 9785, 9788, 9791, 9792, 9794, 9795, 9796, 9797, 9798, 9799, 9799, 9800, 9802, 9805, 9805, 9805, 9807, 9807, 9809, 9811, 9813, 9817, 9817, 9819, 9819, 9819, 9819, 9820, 9821, 9821, 9822, 9825, 9825, 9825, 9825, 9826, 9828, 9828, 9828, 9829, 9831, 9831, 9832, 9833, 9834, 9834, 9834, 9835, 9835, 9837, 9837, 9837, 9837, 9837, 9839, 9840, 9840, 9842, 9843, 9844, 9844, 9845, 9845, 9847, 9847, 9850, 9850, 9855, 9856, 9856, 9856, 9856, 9856, 9856, 9860, 9860, 9861, 9862, 9865, 9865, 9866, 9867, 9868, 9868, 9869, 9870, 9870, 9874, 9874, 9875, 9877, 9878, 9879, 9879, 9881, 9881, 9881, 9882, 9882, 9883, 9884, 9885, 9885, 9886, 9886, 9889, 9889, 9889, 9896, 9898, 9899, 9899, 9899, 9900, 9900, 9916, 9917, 9921, 9921, 9922, 9922, 9928, 9929, 9935, 9935, 9935, 9937, 9937, 9938, 9940, 9943, 9944, 9946, 9946, 9946, 9948, 9951, 9951, 9952, 9957, 9957, 9959, 9960, 9963, 9963, 9966, 9972, 9973, 9974, 9976, 9979, 9984, 9985, 9986, 9986, 9987, 9988, 9989, 9993, 9994, 9994, 9995, 9999, 10000, 10004, 10005, 10005, 10005, 10010, 10012, 10012, 10012, 10014, 10017, 10017, 10018, 10020, 10021, 10022, 10025, 10025, 10029, 10029, 10030, 10030, 10031, 10033, 10038, 10039, 10042, 10044, 10045, 10045, 10048, 10051, 10056, 10057, 10061, 10062, 10062, 10065, 10066, 10068, 10073, 10076, 10077, 10078, 10079, 10081, 10085, 10085, 10085, 10094, 10094, 10095, 10096, 10097, 10097, 10098, 10099, 10099, 10101, 10102, 10108, 10108, 10111, 10111, 10111, 10114, 10114, 10117, 10119, 10121, 10121, 10121, 10121, 10123, 10124, 10124, 10124, 10128, 10131, 10136, 10136, 10141, 10143, 10149, 10149, 10152, 10154, 10155, 10155, 10155, 10156, 10159, 10164, 10164, 10166, 10167, 10172, 10173, 10174, 10175, 10176, 10178, 10178, 10178, 10181, 10182, 10185, 10189, 10189, 10189, 10190, 10190, 10191, 10199, 10200, 10200, 10205, 10206, 10207, 10208, 10208, 10208, 10212, 10214, 10218, 10221, 10223, 10224, 10230, 10231, 10231, 10237, 10237, 10238, 10239, 10240, 10242, 10244, 10244, 10245, 10245, 10245, 10245, 10245, 10245, 10245, 10246, 10248, 10248, 10251, 10251, 10253, 10258, 10259, 10260, 10261, 10261, 10264, 10265, 10266, 10266, 10266, 10270, 10271, 10272, 10274, 10275, 10275, 10276, 10276, 10278, 10279, 10280, 10282, 10284, 10284, 10284, 10286, 10292, 10295, 10299, 10301, 10303, 10303, 10304, 10304, 10304, 10304, 10304, 10304, 10304, 10304, 10305, 10305, 10306, 10306, 10306, 10307, 10307, 10308, 10308, 10308, 10309, 10311, 10311, 10312, 10312, 10313, 10313, 10314, 10317, 10319, 10323, 10324, 10329, 10329, 10331, 10331, 10331, 10331, 10334, 10334, 10335, 10335, 10335, 10335, 10338, 10338, 10341, 10341, 10342, 10344, 10346, 10347, 10351, 10351, 10351, 10353, 10354, 10354, 10355, 10355, 10355, 10355, 10355, 10355, 10355, 10355, 10357, 10357, 10358, 10358, 10358, 10359, 10360, 10360, 10360, 10361, 10362, 10370, 10371, 10374, 10375, 10375, 10375, 10377, 10378, 10378, 10378, 10378, 10379, 10379, 10379, 10380, 10381, 10383, 10385, 10386, 10387, 10387, 10389, 10390, 10390, 10391, 10391, 10396, 10396, 10397, 10398, 10399, 10400, 10402, 10402, 10403, 10403, 10403, 10404, 10405, 10408, 10408, 10409, 10409, 10410, 10412, 10412, 10412, 10412, 10412, 10414, 10416, 10418, 10418, 10418, 10419, 10421, 10421, 10422, 10422, 10422, 10423, 10424, 10424, 10428, 10431, 10432, 10433, 10438, 10441, 10441, 10443, 10444, 10445, 10446, 10446, 10446, 10448, 10448, 10449, 10450, 10450, 10450, 10451, 10451, 10453, 10453, 10454, 10454, 10454, 10455, 10455, 10455, 10455, 10455, 10455, 10455, 10456, 10456, 10457, 10457, 10457, 10458, 10459, 10459, 10460, 10462, 10464, 10467, 10467, 10468, 10468, 10469, 10469, 10470, 10470, 10470, 10472, 10474, 10475, 10476, 10476, 10476, 10476, 10477, 10477, 10478, 10478, 10478, 10479, 10480, 10481, 10481, 10481, 10481, 10482, 10482, 10483, 10483, 10484, 10485, 10485, 10485, 10485, 10487, 10488, 10489, 10490, 10490, 10491, 10492, 10492, 10493, 10496, 10496, 10496, 10497, 10497, 10497, 10497, 10497, 10497, 10498, 10498, 10499, 10500, 10502, 10502, 10502, 10504, 10504, 10504, 10505, 10505, 10505, 10508, 10509, 10509, 10509, 10511, 10511, 10512, 10512, 10514, 10515, 10517, 10517, 10518, 10518, 10519, 10520, 10521, 10522, 10522, 10522, 10523, 10523, 10524, 10524, 10524, 10526, 10527, 10527, 10529, 10530, 10531, 10531, 10531, 10531, 10531, 10532, 10533, 10536, 10536, 10537, 10539, 10540, 10541, 10541, 10541, 10541, 10541, 10543, 10545, 10545, 10545, 10547, 10548, 10548, 10551, 10553, 10553, 10554, 10554, 10555, 10556, 10558, 10565, 10566, 10570, 10570, 10572, 10572, 10574, 10574, 10575, 10576, 10578, 10580, 10581, 10584, 10584, 10584, 10584, 10586, 10587, 10588, 10591, 10591, 10591, 10594, 10595, 10595, 10595, 10595, 10595, 10595, 10595, 10595, 10596, 10596, 10597, 10597, 10597, 10597, 10598, 10599, 10601, 10603, 10609, 10609, 10609, 10610, 10610, 10613, 10614, 10616, 10617, 10619, 10619, 10621, 10621, 10622, 10623, 10624, 10628, 10630, 10643, 10644, 10647, 10650, 10651, 10657, 10657, 10657, 10658, 10658, 10660, 10662, 10664, 10665, 10666, 10667, 10669, 10669, 10671, 10673, 10673, 10674, 10678, 10680, 10682, 10683, 10686, 10686, 10686, 10688, 10689, 10691, 10691, 10691, 10691, 10696, 10696, 10698, 10699, 10699, 10701, 10701, 10703, 10706, 10707, 10707, 10710, 10711, 10712, 10713, 10715, 10715, 10715, 10716, 10717, 10717, 10718, 10719, 10719, 10722, 10722, 10724, 10724, 10724, 10725, 10725, 10725, 10725, 10726, 10726, 10727, 10727, 10728, 10729, 10730, 10731, 10733, 10733, 10733, 10734, 10734, 10735, 10735, 10737, 10737, 10737, 10739, 10741, 10741, 10742, 10742, 10742, 10742, 10743, 10745, 10747, 10747, 10747, 10752, 10756, 10760, 10761, 10763, 10763, 10764, 10765, 10766, 10767, 10768, 10769, 10770, 10771, 10771, 10772, 10773, 10773, 10774, 10774, 10775, 10776, 10777, 10777, 10778, 10781, 10782, 10784, 10785, 10786, 10786, 10787, 10787, 10788, 10789, 10790, 10790, 10793, 10793, 10793, 10795, 10796, 10797, 10797, 10797, 10797, 10797, 10798, 10798, 10800, 10800, 10801, 10802, 10802, 10803, 10804, 10805, 10805, 10805, 10806, 10806, 10807, 10807, 10809, 10809, 10809, 10813, 10813, 10814, 10814, 10814, 10815, 10815, 10816, 10816, 10816, 10818, 10819, 10823, 10823, 10827, 10828, 10829, 10829, 10830, 10830, 10831, 10831, 10831, 10832, 10833, 10836, 10836, 10839, 10840, 10842, 10843, 10844, 10845, 10846, 10848, 10848, 10853, 10853, 10856, 10856, 10857, 10857, 10858, 10859, 10859, 10859, 10859, 10861, 10862, 10862, 10864, 10864, 10864, 10866, 10866, 10866, 10866, 10869, 10870, 10872, 10872, 10878, 10880, 10880, 10882, 10883, 10883, 10883, 10884, 10886, 10889, 10889, 10891, 10891, 10891, 10897, 10897, 10897, 10898, 10898, 10898, 10898, 10899, 10900, 10901, 10901, 10902, 10902, 10904, 10906, 10907, 10907, 10910, 10913, 10913, 10914, 10914, 10914, 10914, 10914, 10915, 10916, 10917, 10919, 10921, 10921, 10921, 10921, 10924, 10924, 10924, 10925, 10925, 10925, 10928, 10929, 10931, 10933, 10933, 10935, 10936, 10936, 10937, 10937, 10938, 10938, 10938, 10938, 10938, 10940, 10940, 10940, 10941, 10941, 10941, 10942, 10942, 10950, 10950, 10953, 10959, 10959, 10960, 10961, 10961, 10962, 10962, 10963, 10963, 10964, 10964, 10969, 10971, 10971, 10971, 10973, 10973, 10973, 10973, 10973, 10975, 10977, 10977, 10980, 10983, 10983, 10983, 10985, 10990, 10990, 10992, 10992, 10997, 10998, 11003, 11004, 11010, 11011, 11011, 11013, 11013, 11014, 11014, 11015, 11019, 11020, 11026, 11027, 11028, 11033, 11033, 11036, 11036, 11036, 11036, 11036, 11036, 11036, 11036, 11038, 11040, 11040, 11041, 11041, 11044, 11045, 11047, 11048, 11048, 11053, 11054, 11056, 11056, 11057, 11057, 11057, 11058, 11058, 11060, 11061, 11063, 11065, 11065, 11067, 11068, 11071, 11072, 11072, 11072, 11072, 11073, 11074, 11078, 11080, 11080, 11082, 11085, 11086, 11088, 11089, 11090, 11093, 11094, 11095, 11096, 11097, 11100, 11101, 11101, 11104, 11104, 11104, 11105, 11106, 11109, 11109, 11109, 11109, 11113, 11114, 11114, 11114, 11114, 11122, 11122, 11122, 11124, 11125, 11126, 11126, 11128, 11128, 11128, 11128, 11130, 11130, 11130, 11132, 11133, 11134, 11134, 11134, 11134, 11136, 11136, 11137, 11138, 11141, 11142, 11143, 11145, 11145, 11145, 11149, 11150, 11152, 11152, 11152, 11152, 11153, 11153, 11155, 11155, 11155, 11155, 11156, 11156, 11156, 11156, 11156, 11158, 11159, 11159, 11162, 11165, 11169, 11171, 11172, 11174, 11175, 11178, 11179, 11180, 11180, 11180, 11180, 11180, 11180, 11182, 11183, 11183, 11183, 11185, 11185, 11186, 11186, 11187, 11188, 11188, 11188, 11188, 11188, 11188, 11189, 11189, 11190, 11191, 11191, 11198, 11199, 11202, 11203, 11204, 11206, 11207, 11208, 11208, 11208, 11211, 11211, 11212, 11212, 11212, 11212, 11213, 11214, 11214, 11216, 11218, 11219, 11222, 11222, 11222, 11223, 11223, 11224, 11225, 11228, 11229, 11230, 11231, 11232, 11232, 11238, 11239, 11239, 11239, 11241, 11242, 11244, 11246, 11247, 11250, 11250, 11250, 11250, 11251, 11252, 11252, 11255, 11256, 11260, 11260, 11260, 11261, 11263, 11263, 11263, 11264, 11264, 11265, 11271, 11273, 11274, 11277, 11279, 11279, 11279, 11281, 11283, 11283, 11283, 11286, 11288, 11288, 11289, 11290, 11290, 11290, 11292, 11297, 11299, 11299, 11301, 11301, 11304, 11304, 11305, 11305, 11308, 11311, 11312, 11315, 11316, 11317, 11320, 11320, 11323, 11325, 11328, 11333, 11333, 11334, 11334, 11336, 11337, 11339, 11340, 11342, 11342, 11343, 11344, 11346, 11348, 11348, 11348, 11351, 11353, 11353, 11354, 11358, 11358, 11360, 11360, 11360, 11360, 11363, 11363, 11368, 11368, 11371, 11371, 11381, 11381, 11381, 11383, 11386, 11388, 11389, 11392, 11392, 11393, 11394, 11395, 11395, 11396, 11398, 11398, 11398, 11398, 11400, 11400, 11401, 11401, 11401, 11401, 11402, 11404, 11404, 11404, 11412, 11413, 11413, 11417, 11417, 11417, 11422, 11428, 11429, 11430, 11431, 11432, 11433, 11435, 11436, 11436, 11438, 11446, 11452, 11452, 11453, 11453, 11453, 11455, 11462, 11466, 11467, 11469, 11470, 11471, 11471, 11471, 11471, 11472, 11472, 11473, 11475, 11482, 11483, 11483, 11484, 11485, 11488, 11489, 11489, 11493, 11493, 11496, 11496, 11496, 11497, 11500, 11502, 11502, 11503, 11503, 11504, 11504, 11506, 11507, 11508, 11510, 11512, 11513, 11513, 11514, 11515, 11519, 11520, 11520, 11521, 11523, 11527, 11528, 11529, 11530, 11531, 11532, 11534, 11534, 11534, 11534, 11536, 11543, 11544, 11544, 11544, 11546, 11548, 11548, 11550, 11551, 11552, 11552, 11552, 11553, 11554, 11554, 11557, 11557, 11559, 11559, 11559, 11561, 11565, 11569, 11570, 11572, 11573, 11576, 11576, 11577, 11577, 11585, 11585, 11586, 11586, 11586, 11587, 11588, 11588, 11589, 11590, 11605, 11605, 11607, 11609, 11610, 11610, 11611, 11613, 11618, 11619, 11621, 11622, 11623, 11623, 11623, 11624, 11624, 11624, 11625, 11625, 11626, 11626, 11627, 11628, 11628, 11630, 11630, 11631, 11632, 11633, 11633, 11636, 11636, 11636, 11636, 11637, 11637, 11638, 11638, 11639, 11641, 11642, 11643, 11643, 11644, 11647, 11649, 11649, 11653, 11655, 11659, 11661, 11661, 11662, 11663, 11667, 11667, 11667, 11672, 11674, 11674, 11674, 11675, 11675, 11677, 11679, 11682, 11682, 11683, 11684, 11686, 11686, 11689, 11690, 11691, 11696, 11699, 11699, 11699, 11700, 11700, 11701, 11703, 11703, 11705, 11706, 11708, 11709, 11709, 11711, 11711, 11712, 11713, 11716, 11718, 11718, 11718, 11720, 11723, 11724, 11727, 11728, 11728, 11732, 11734, 11734, 11739, 11742, 11743, 11743, 11743, 11744, 11744, 11744, 11746, 11748, 11750, 11753, 11754, 11754, 11757, 11761, 11761, 11762, 11763, 11763, 11768, 11770, 11771, 11776, 11781, 11781, 11781, 11781, 11787, 11791, 11792, 11792, 11792, 11793, 11793, 11795, 11796, 11796, 11798, 11798, 11800, 11800, 11801, 11802, 11805, 11806, 11806, 11807, 11808, 11808, 11808, 11810, 11811, 11811, 11812, 11814, 11814, 11814, 11818, 11819, 11823, 11825, 11826, 11826, 11827, 11829, 11830, 11830, 11832, 11833, 11833, 11833, 11834, 11834, 11835, 11835, 11840, 11842, 11842, 11842, 11843, 11843, 11843, 11845, 11846, 11847, 11849, 11850, 11850, 11850, 11851, 11852, 11854, 11855, 11855, 11857, 11857, 11860, 11861, 11864, 11867, 11867, 11867, 11867, 11867, 11868, 11869, 11869, 11870, 11871, 11871, 11871, 11875, 11876, 11876, 11876, 11877, 11880, 11880, 11884, 11885, 11888, 11890, 11890, 11893, 11897, 11897, 11898, 11901, 11902, 11905, 11905, 11908, 11910, 11911, 11916, 11918, 11919, 11921, 11921, 11928, 11930, 11930, 11930, 11932, 11939, 11940, 11941, 11945, 11950, 11950, 11954, 11955, 11955, 11955, 11959, 11959, 11959, 11960, 11962, 11964, 11966, 11969, 11975, 11976, 11979, 11980, 11985, 11985, 11990, 11990, 11992, 11998, 11999, 12001, 12001, 12003, 12004, 12007, 12007, 12010, 12020, 12020, 12023, 12025, 12027, 12029, 12033, 12033, 12033, 12033, 12036, 12037, 12038, 12038, 12040, 12041, 12043, 12045, 12047, 12051, 12054, 12055, 12061, 12061, 12062, 12062, 12063, 12063, 12063, 12064, 12064, 12064, 12066, 12066, 12068, 12068, 12069, 12069, 12069, 12070, 12070, 12071, 12071, 12073, 12075, 12077, 12081, 12081, 12082, 12083, 12083, 12085, 12085, 12085, 12087, 12091, 12092, 12096, 12097, 12098, 12098, 12101, 12101, 12101, 12101, 12103, 12104, 12105, 12107, 12109, 12111, 12111, 12112, 12112, 12112, 12112, 12114, 12115, 12115, 12116, 12118, 12119, 12124, 12125, 12126, 12130, 12130, 12130, 12130, 12130, 12130, 12132, 12132, 12135, 12136, 12137, 12138, 12139, 12139, 12140, 12140, 12140, 12140, 12142, 12143, 12143, 12143, 12144, 12144, 12144, 12148, 12149, 12149, 12152, 12152, 12154, 12154, 12155, 12155, 12157, 12159, 12159, 12163, 12166, 12169, 12172, 12179, 12179, 12181, 12183, 12185, 12189, 12190, 12190, 12194, 12194, 12195, 12195, 12195, 12196, 12196, 12196, 12199, 12200, 12203, 12204, 12206, 12206, 12206, 12206, 12206, 12207, 12208, 12210, 12213, 12213, 12215, 12215, 12215, 12215, 12219, 12221, 12224, 12225, 12225, 12226, 12229, 12229, 12236, 12237, 12237, 12239, 12239, 12239, 12246, 12246, 12249, 12249, 12250, 12250, 12250, 12251, 12254, 12254, 12254, 12255, 12256, 12257, 12260, 12262, 12265, 12265, 12266, 12269, 12269, 12270, 12277, 12278, 12279, 12285, 12285, 12285, 12288, 12289, 12294, 12294, 12294, 12295, 12295, 12303, 12307, 12308, 12311, 12315, 12315, 12316, 12318, 12321, 12322, 12327, 12328, 12329, 12329, 12329, 12329, 12332, 12332, 12333, 12335, 12335, 12335, 12338, 12338, 12338, 12338, 12338, 12338, 12339, 12341, 12343, 12344, 12344, 12345, 12345, 12346, 12347, 12349, 12351, 12352, 12352, 12352, 12352, 12352, 12352, 12353, 12353, 12356, 12356, 12359, 12359, 12360, 12361, 12362, 12363, 12363, 12365, 12366, 12366, 12366, 12366, 12366, 12367, 12367, 12368, 12368, 12370, 12370, 12371, 12371, 12374, 12375, 12375, 12377, 12378, 12379, 12381, 12383, 12384, 12384, 12384, 12384, 12384, 12385, 12388, 12388, 12388, 12388, 12393, 12394, 12402, 12403, 12403, 12403, 12403, 12405, 12406, 12407, 12407, 12407, 12408, 12411, 12412, 12412, 12412, 12412, 12412, 12412, 12413, 12415, 12415, 12420, 12421, 12423, 12423, 12425, 12427, 12428, 12429, 12430, 12435, 12437, 12437, 12437, 12444, 12444, 12446, 12446, 12449, 12449, 12450, 12450, 12450, 12452, 12454, 12459, 12459, 12460, 12460, 12460, 12460, 12461, 12461, 12463, 12466, 12466, 12468, 12468, 12472, 12475, 12475, 12477, 12478, 12485, 12486, 12487, 12488, 12489, 12493, 12494, 12494, 12495, 12495, 12497, 12497, 12497, 12497, 12497, 12499, 12500, 12502, 12502, 12502, 12504, 12505, 12506, 12511, 12516, 12517, 12522, 12522, 12522, 12523, 12526, 12527, 12527, 12527, 12528, 12530, 12530, 12533, 12533, 12534, 12538, 12539, 12540, 12545, 12545, 12548, 12549, 12551, 12551, 12555, 12555, 12557, 12558, 12559, 12561, 12561, 12569, 12570, 12572, 12578, 12580, 12581, 12581, 12581, 12583, 12583, 12589, 12589, 12591, 12592, 12595, 12596, 12596, 12598, 12598, 12600, 12601, 12601, 12602, 12602, 12602, 12603, 12605, 12605, 12605, 12605, 12608, 12610, 12610, 12612, 12615, 12615, 12615, 12615, 12615, 12615, 12616, 12616, 12617, 12617, 12617, 12617, 12618, 12618, 12618, 12621, 12621, 12621, 12622, 12623, 12624, 12625, 12625, 12625, 12626, 12626, 12626, 12628, 12628, 12628, 12629, 12629, 12631, 12631, 12633, 12634, 12635, 12635, 12635, 12635, 12635, 12638, 12638, 12638, 12640, 12640, 12641, 12641, 12643, 12644, 12644, 12644, 12644, 12645, 12646, 12646, 12646, 12646, 12646, 12646, 12646, 12648, 12648, 12649, 12650, 12650, 12650, 12650, 12650, 12651, 12654, 12654, 12656, 12656, 12658, 12659, 12660, 12660, 12662, 12662, 12663, 12663, 12663, 12664, 12665, 12665, 12666, 12667, 12667, 12667, 12668, 12669, 12669, 12669, 12670, 12673, 12673, 12674, 12675, 12675, 12675, 12675, 12675, 12676, 12677, 12677, 12677, 12677, 12677, 12677, 12678, 12679, 12680, 12680, 12680, 12681, 12681, 12681, 12681, 12681, 12682, 12682, 12682, 12682, 12683, 12683, 12685, 12685, 12685, 12685, 12685, 12687, 12687, 12687, 12687, 12687, 12688, 12688, 12688, 12689, 12690, 12691, 12691, 12691, 12691, 12692, 12693, 12693, 12694, 12695, 12695, 12696, 12696, 12696, 12696, 12697, 12697, 12697, 12697, 12697, 12698, 12698, 12698, 12698, 12699, 12699, 12700, 12700, 12701, 12703, 12703, 12705, 12708, 12708, 12713, 12716, 12717, 12721, 12722, 12722, 12725, 12728, 12729, 12732, 12733, 12733, 12733, 12734, 12735, 12739, 12739, 12739, 12739, 12740, 12740, 12740, 12744, 12747, 12747, 12747, 12747, 12747, 12747, 12749, 12751, 12752, 12754, 12760, 12762, 12764, 12774, 12774, 12775, 12775, 12776, 12779, 12779, 12779, 12779, 12781, 12782, 12782, 12782, 12784, 12786, 12786, 12786, 12786, 12786, 12788, 12794, 12795, 12800, 12801, 12801, 12805, 12806, 12806, 12806, 12806, 12808, 12808, 12809, 12811, 12814, 12817, 12817, 12819, 12819, 12819, 12819, 12821, 12826, 12827, 12829, 12830, 12830, 12830, 12830, 12833, 12834, 12836, 12837, 12839, 12839, 12839, 12839, 12840, 12841, 12843, 12844, 12844, 12845, 12845, 12845, 12846, 12847, 12847, 12847, 12850, 12850, 12850, 12854, 12854, 12854, 12855, 12855, 12855, 12856, 12857, 12860, 12860, 12861, 12861, 12862, 12863, 12864, 12865, 12869, 12871, 12871, 12871, 12871, 12872, 12873, 12873, 12873, 12877, 12878, 12879, 12885, 12887, 12888, 12890, 12893, 12893, 12893, 12899, 12901, 12906, 12908, 12909, 12909, 12913, 12913, 12918, 12918, 12921, 12927, 12927, 12929, 12929, 12932, 12933, 12933, 12935, 12936, 12939, 12940, 12940, 12945, 12946, 12947, 12947, 12955, 12957, 12960, 12962, 12962, 12962, 12972, 12972, 12976, 12977, 12978, 12979, 12981, 12981, 12990, 12992, 12995, 12996, 12996, 12996, 12996, 12999, 13001, 13006, 13006, 13006, 13007, 13009, 13014, 13016, 13017, 13017, 13018, 13020, 13022, 13023, 13025, 13025, 13027, 13027, 13027, 13028, 13033, 13033, 13034, 13034, 13034, 13035, 13035, 13036, 13038, 13038, 13038, 13039, 13041, 13042, 13044, 13047, 13053, 13055, 13055, 13067, 13067, 13067, 13067, 13067, 13069, 13071, 13078, 13082, 13082, 13083, 13083, 13083, 13084, 13087, 13089, 13089, 13091, 13092, 13093, 13093, 13094, 13097, 13098, 13099, 13107, 13111, 13111, 13111, 13112, 13115, 13116, 13118, 13118, 13118, 13120, 13123, 13128, 13129, 13129, 13130, 13133, 13133, 13142, 13144, 13146, 13146, 13148, 13150, 13151, 13156, 13157, 13157, 13158, 13160, 13160, 13160, 13161, 13161, 13162, 13162, 13164, 13167, 13167, 13169, 13170, 13171, 13180, 13181, 13182, 13184, 13185, 13185, 13185, 13186, 13186, 13186, 13186, 13189, 13189, 13191, 13194, 13194, 13194, 13197, 13199, 13201, 13203, 13203, 13204, 13209, 13210, 13210, 13210, 13210, 13216, 13218, 13221, 13224, 13224, 13225, 13225, 13225, 13225, 13231, 13231, 13232, 13232, 13233, 13233, 13236, 13237, 13237, 13237, 13239, 13240, 13244, 13245, 13248, 13253, 13255, 13256, 13256, 13257, 13259, 13261, 13262, 13262, 13263, 13264, 13266, 13266, 13267, 13269, 13271, 13277, 13281, 13282, 13282, 13282, 13285, 13285, 13286, 13286, 13286, 13290, 13290, 13290, 13291, 13292, 13295, 13295, 13295, 13296, 13296, 13299, 13302, 13302, 13303, 13304, 13305, 13306, 13306, 13307, 13307, 13310, 13315, 13320, 13320, 13320, 13322, 13324, 13326, 13331, 13332, 13332, 13332, 13333, 13334, 13334, 13335, 13337, 13338, 13339, 13339, 13339, 13340, 13341, 13341, 13342, 13344, 13344, 13345, 13347, 13347, 13347, 13349, 13351, 13351, 13351, 13352, 13353, 13353, 13353, 13355, 13356, 13356, 13357, 13357, 13358, 13361, 13361, 13361, 13362, 13362, 13365, 13366, 13367, 13368, 13369, 13369, 13370, 13372, 13373, 13374, 13374, 13377, 13379, 13379, 13379, 13379, 13379, 13380, 13380, 13380, 13380, 13380, 13384, 13385, 13385, 13385, 13386, 13386, 13386, 13387, 13387, 13387, 13387, 13387, 13389, 13389, 13393, 13393, 13394, 13395, 13395, 13397, 13397, 13398, 13401, 13401, 13401, 13401, 13403, 13403, 13403, 13404, 13405, 13406, 13407, 13409, 13409, 13411, 13411, 13413, 13413, 13414, 13414, 13414, 13414, 13414, 13414, 13415, 13415, 13415, 13416, 13416, 13417, 13417, 13417, 13419, 13419, 13422, 13422, 13422, 13422, 13423, 13423, 13424, 13425, 13425, 13425, 13425, 13425, 13426, 13428, 13429, 13430, 13430, 13431, 13431, 13432, 13433, 13433, 13434, 13437, 13439, 13439, 13439, 13449, 13451, 13459, 13459, 13459, 13460, 13461, 13462, 13463, 13464, 13464, 13465, 13465, 13467, 13470, 13472, 13472, 13475, 13476, 13476, 13477, 13479, 13480, 13484, 13487, 13487, 13489, 13489, 13490, 13493, 13495, 13495, 13496, 13497, 13498, 13501, 13501, 13503, 13505, 13505, 13506, 13507, 13508, 13509, 13509, 13510, 13510, 13510, 13513, 13514, 13515, 13515, 13516, 13517, 13517, 13517, 13518, 13518, 13519, 13520, 13520, 13520, 13521, 13521, 13522, 13523, 13523, 13525, 13526, 13527, 13528, 13528, 13533, 13538, 13539, 13540, 13543, 13543, 13544, 13544, 13544, 13544, 13545, 13545, 13548, 13549, 13550, 13551, 13555, 13556, 13556, 13557, 13559, 13564, 13565, 13566, 13566, 13566, 13566, 13568, 13568, 13571, 13571, 13572, 13572, 13573, 13573, 13573, 13573, 13573, 13573, 13574, 13574, 13574, 13575, 13577, 13580, 13581, 13581, 13581, 13581, 13581, 13584, 13584, 13584, 13584, 13589, 13590, 13592, 13592, 13594, 13594, 13595, 13596, 13597, 13597, 13597, 13598, 13598, 13601, 13603, 13605, 13609, 13610, 13610, 13614, 13616, 13617, 13617, 13617, 13619, 13621, 13623, 13624, 13626, 13627, 13627, 13628, 13631, 13631, 13632, 13632, 13633, 13636, 13639, 13639, 13641, 13642, 13643, 13643, 13644, 13644, 13644, 13645, 13645, 13645, 13647, 13649, 13650, 13653, 13656, 13660, 13660, 13661, 13661, 13662, 13662, 13662, 13662, 13663, 13663, 13663, 13663, 13663, 13664, 13671, 13675, 13675, 13675, 13676, 13677, 13678, 13678, 13678, 13678, 13678, 13680, 13685, 13686, 13687, 13687, 13689, 13690, 13690, 13694, 13694, 13700, 13700, 13702, 13702, 13702, 13704, 13705, 13708, 13708, 13717, 13721, 13722, 13726, 13727, 13731, 13732, 13733, 13735, 13736, 13736, 13738, 13741, 13741, 13743, 13745, 13748, 13750, 13750, 13752, 13758, 13758, 13758, 13758, 13760, 13761, 13763, 13765, 13766, 13769, 13771, 13773, 13773, 13773, 13776, 13778, 13779, 13780, 13789, 13791, 13794, 13798, 13800, 13803, 13804, 13812, 13816, 13816, 13816, 13816, 13821, 13822, 13822, 13827, 13827, 13828, 13829, 13831, 13832, 13832, 13832, 13834, 13841, 13841, 13842, 13842, 13846, 13848, 13851, 13855, 13855, 13860, 13861, 13864, 13865, 13866, 13866, 13871, 13871, 13871, 13874, 13875, 13878, 13880, 13882, 13882, 13882, 13882, 13886, 13887, 13889, 13890, 13890, 13891, 13892, 13892, 13892, 13894, 13894, 13897, 13898, 13899, 13900, 13900, 13902, 13902, 13902, 13903, 13903, 13903, 13903, 13903, 13904, 13904, 13907, 13907, 13912, 13915, 13916, 13916, 13916, 13919, 13919, 13920, 13920, 13921, 13921, 13924, 13925, 13925, 13926, 13926, 13927, 13928, 13930, 13932, 13933, 13934, 13936, 13936, 13939, 13939, 13939, 13943, 13943, 13943, 13944, 13945, 13946, 13946, 13946, 13946, 13948, 13952, 13953, 13953, 13953, 13954, 13955, 13955, 13955, 13955, 13955, 13959, 13959, 13959, 13962, 13966, 13966, 13966, 13966, 13967, 13968, 13968, 13968, 13969, 13970, 13970, 13972, 13973, 13973, 13973, 13974, 13974, 13976, 13976, 13976, 13976, 13976, 13980, 13981, 13982, 13983, 13983, 13983, 13984, 13988, 13988, 13988, 13989, 13989, 13989, 13989, 13989, 13989, 13990, 13991, 13992, 13993, 13993, 13994, 13995, 13996, 13996, 14001, 14003, 14004, 14006, 14013, 14013, 14015, 14015, 14015, 14017, 14018, 14018, 14018, 14019, 14019, 14021, 14021, 14021, 14021, 14021, 14023, 14025, 14026, 14026, 14026, 14026, 14029, 14029, 14029, 14029, 14038, 14038, 14038, 14040, 14040, 14040, 14044, 14049, 14050, 14050, 14052, 14053, 14056, 14059, 14059, 14059, 14060, 14063, 14067, 14067, 14069, 14071, 14073, 14074, 14076, 14077, 14078, 14078, 14080, 14081, 14082, 14082, 14083, 14084, 14084, 14085, 14087, 14089, 14089, 14090, 14091, 14091, 14092, 14095, 14095, 14099, 14099, 14099, 14103, 14105, 14107, 14108, 14108, 14111, 14112, 14113, 14113, 14113, 14114, 14115, 14117, 14118, 14118, 14118, 14118, 14119, 14120, 14121, 14122, 14123, 14123, 14125, 14125, 14126, 14128, 14133, 14135, 14136, 14140, 14140, 14141, 14143, 14146, 14147, 14150, 14152, 14153, 14153, 14154, 14154, 14156, 14157, 14160, 14160, 14161, 14164, 14168, 14169, 14172, 14172, 14174, 14174, 14174, 14174, 14175, 14176, 14177, 14177, 14181, 14182, 14182, 14182, 14188, 14189, 14191, 14191, 14191, 14191, 14192, 14192, 14192, 14193, 14193, 14194, 14197, 14198, 14198, 14200, 14201, 14202, 14202, 14202, 14210, 14211, 14211, 14212, 14213, 14214, 14215, 14220, 14220, 14220, 14225, 14226, 14226, 14237, 14237, 14237, 14239, 14240, 14242, 14244, 14246, 14246, 14248, 14248, 14249, 14249, 14250, 14255, 14255, 14257, 14266, 14266, 14266, 14267, 14267, 14267, 14269, 14269, 14269, 14270, 14270, 14270, 14270, 14271, 14272, 14272, 14277, 14277, 14279, 14281, 14281, 14281, 14281, 14285, 14289, 14292, 14295, 14295, 14298, 14299, 14300, 14300, 14301, 14302, 14302, 14303, 14308, 14310, 14311, 14312, 14312, 14316, 14316, 14316, 14316, 14318, 14318, 14320, 14321, 14324, 14327, 14332, 14332, 14336, 14338, 14338, 14338, 14338, 14338, 14339, 14339, 14339, 14343, 14349, 14352, 14352, 14355, 14357, 14357, 14360, 14360, 14361, 14361, 14362, 14363, 14363, 14368, 14371, 14372, 14372, 14373, 14373, 14373, 14376, 14376, 14377, 14381, 14383, 14384, 14385, 14385, 14385, 14386, 14386, 14386, 14389, 14393, 14393, 14393, 14393, 14393, 14396, 14396, 14397, 14403, 14403, 14403, 14403, 14403, 14404, 14405, 14406, 14406, 14410, 14410, 14410, 14412, 14413, 14416, 14417, 14418, 14421, 14422, 14422, 14423, 14424, 14425, 14426, 14428, 14428, 14429, 14431, 14431, 14433, 14435, 14437, 14439, 14441, 14441, 14443, 14443, 14445, 14446, 14447, 14450, 14450, 14453, 14453, 14453, 14453, 14454, 14454, 14455, 14458, 14458, 14459, 14460, 14460, 14465, 14466, 14466, 14467, 14467, 14468, 14468, 14469, 14478, 14480, 14481, 14481, 14483, 14485, 14488, 14488, 14491, 14491, 14492, 14492, 14493, 14493, 14494, 14494, 14495, 14496, 14497, 14497, 14499, 14500, 14500, 14501, 14511, 14511, 14512, 14514, 14514, 14514, 14516, 14517, 14517, 14519, 14519, 14519, 14519, 14521, 14523, 14524, 14526, 14526, 14527, 14528, 14529, 14532, 14540, 14540, 14543, 14543, 14546, 14548, 14548, 14549, 14552, 14552, 14552, 14555, 14556, 14558, 14558, 14559, 14564, 14564, 14565, 14565, 14565, 14566, 14566, 14573, 14575, 14575, 14581, 14581, 14581, 14586, 14587, 14587, 14590, 14592, 14592, 14594, 14595, 14597, 14599, 14603, 14604, 14608, 14612, 14612, 14614, 14614, 14617, 14617, 14617, 14617, 14620, 14623, 14623, 14625, 14632, 14633, 14633, 14635, 14635, 14637, 14638, 14639, 14641, 14643, 14643, 14644, 14647, 14651, 14652, 14654, 14654, 14654, 14655, 14656, 14658, 14663, 14667, 14669, 14669, 14671, 14677, 14677, 14679, 14681, 14681, 14681, 14685, 14685, 14688, 14689, 14691, 14691, 14692, 14693, 14694, 14695, 14695, 14698, 14699, 14699, 14699, 14704, 14704, 14705, 14712, 14719, 14720, 14721, 14722, 14724, 14724, 14725, 14725, 14725, 14728, 14730, 14731, 14733, 14733, 14737, 14738, 14739, 14740, 14740, 14741, 14742, 14742, 14747, 14747, 14752, 14754, 14755, 14757, 14757, 14758, 14758, 14758, 14759, 14759, 14759, 14760, 14761, 14761, 14763, 14764, 14768, 14769, 14771, 14771, 14774, 14774, 14774, 14778, 14778, 14778, 14778, 14779, 14782, 14783, 14785, 14787, 14788, 14793, 14796, 14797, 14797, 14798, 14801, 14802, 14804, 14805, 14805, 14814, 14814, 14816, 14816, 14816, 14817, 14817, 14819, 14822, 14826, 14830, 14831, 14833, 14834, 14834, 14836, 14837, 14837, 14837, 14839, 14839, 14839, 14839, 14840, 14840, 14840, 14842, 14847, 14848, 14849, 14849, 14850, 14854, 14864, 14867, 14868, 14869, 14873, 14874, 14874, 14874, 14874, 14874, 14875, 14875, 14879, 14880, 14880, 14881, 14881, 14881, 14882, 14883, 14886, 14886, 14886, 14886, 14892, 14893, 14895, 14898, 14903, 14904, 14906, 14910, 14910, 14910, 14910, 14916, 14916, 14918, 14919, 14921, 14923, 14923, 14923, 14923, 14924, 14924, 14924, 14929, 14932, 14933, 14934, 14934, 14935, 14937, 14937, 14938, 14939, 14939, 14939, 14939, 14939, 14940, 14941, 14942, 14942, 14944, 14947, 14949, 14950, 14951, 14953, 14953, 14953, 14954, 14955, 14955, 14956, 14956, 14958, 14959, 14961, 14963, 14965, 14965, 14965, 14965, 14967, 14975, 14976, 14976, 14978, 14979, 14981, 14981, 14982, 14983, 14984, 14986, 14987, 14990, 14992, 14996, 14996, 14996, 14997, 14997, 14998, 14999, 14999, 15002, 15003, 15005, 15006, 15008, 15009, 15011, 15012, 15015, 15019, 15019, 15019, 15020, 15020, 15022, 15022, 15026, 15026, 15028, 15032, 15033, 15033, 15035, 15035, 15035, 15035, 15037, 15038, 15039, 15039, 15039, 15039, 15039, 15041, 15045, 15045, 15046, 15053, 15054, 15057, 15057, 15058, 15058, 15058, 15062, 15063, 15065, 15067, 15067, 15068, 15072, 15074, 15074, 15074, 15078, 15080, 15080, 15080, 15081, 15082, 15082, 15083, 15084, 15085, 15089, 15091, 15091, 15091, 15091, 15093, 15095, 15096, 15096, 15097, 15097, 15098, 15101, 15104, 15105, 15105, 15105, 15105, 15105, 15109, 15111, 15112, 15114, 15115, 15118, 15122, 15127, 15127, 15127, 15127, 15130, 15130, 15136, 15138, 15140, 15141, 15145, 15146, 15147, 15149, 15149, 15151, 15152, 15153, 15154, 15154, 15155, 15155, 15155, 15158, 15158, 15158, 15163, 15164, 15167, 15168, 15169, 15170, 15170, 15171, 15172, 15173, 15177, 15184, 15184, 15186, 15187, 15188, 15190, 15193, 15195, 15197, 15198, 15199, 15199, 15199, 15199, 15199, 15199, 15203, 15203, 15207, 15207, 15207, 15210, 15213, 15213, 15214, 15214, 15214, 15215, 15221, 15223, 15226, 15228, 15229, 15229, 15230, 15232, 15236, 15237, 15239, 15239, 15239, 15241, 15245, 15245, 15245, 15249, 15254, 15254, 15256, 15256, 15256, 15256, 15258, 15258, 15258, 15258, 15259, 15260, 15261, 15263, 15264, 15265, 15266, 15268, 15268, 15270, 15270, 15271, 15271, 15273, 15279, 15280, 15281, 15283, 15283, 15283, 15288, 15289, 15289, 15292, 15300, 15300, 15301, 15301, 15302, 15303, 15308, 15311, 15311, 15314, 15314, 15314, 15315, 15316, 15318, 15318, 15319, 15322, 15326, 15328, 15329, 15331, 15333, 15333, 15334, 15334, 15336, 15338, 15340, 15342, 15342, 15345, 15345, 15346, 15348, 15348, 15348, 15349, 15350, 15357, 15357, 15360, 15360, 15361, 15364, 15366, 15367, 15367, 15368, 15369, 15369, 15370, 15372, 15372, 15373, 15375, 15376, 15376, 15377, 15378, 15378, 15378, 15378, 15378, 15386, 15386, 15387, 15388, 15392, 15393, 15393, 15393, 15393, 15393, 15394, 15394, 15394, 15395, 15396, 15397, 15398, 15400, 15400, 15400, 15400, 15400, 15403, 15403, 15405, 15407, 15410, 15411, 15412, 15414, 15416, 15416, 15417, 15417, 15419, 15422, 15422, 15423, 15423, 15423, 15423, 15425, 15427, 15427, 15429, 15431, 15432, 15433, 15433, 15433, 15433, 15436, 15436, 15437, 15437, 15437, 15437, 15438, 15438, 15439, 15444, 15445, 15445, 15445, 15451, 15451, 15453, 15453, 15454, 15457, 15458, 15459, 15461, 15462, 15462, 15469, 15471, 15471, 15471, 15472, 15477, 15477, 15480, 15480, 15481, 15482, 15482, 15483, 15483, 15484, 15485, 15485, 15487, 15491, 15491, 15494, 15496, 15497, 15502, 15503, 15506, 15507, 15507, 15507, 15508, 15511, 15522, 15522, 15525, 15525, 15526, 15526, 15526, 15532, 15534, 15535, 15536, 15537, 15538, 15538, 15542, 15544, 15546, 15546, 15549, 15550, 15551, 15553, 15554, 15557, 15559, 15560, 15562, 15562, 15562, 15564, 15565, 15566, 15568, 15570, 15570, 15571, 15572, 15572, 15572, 15572, 15574, 15578, 15579, 15579, 15579, 15579, 15580, 15580, 15581, 15581, 15583, 15583, 15585, 15585, 15585, 15587, 15587, 15587, 15589, 15597, 15597, 15603, 15606, 15608, 15608, 15609, 15609, 15609, 15610, 15618, 15620, 15621, 15621, 15622, 15630, 15630, 15632, 15633, 15637, 15637, 15638, 15638, 15638, 15638, 15642, 15643, 15644, 15644, 15645, 15645, 15646, 15647, 15647, 15648, 15650, 15650, 15651, 15652, 15652, 15652, 15652, 15652, 15653, 15653, 15653, 15653, 15653, 15654, 15654, 15654, 15654, 15656, 15656, 15658, 15662, 15662, 15663, 15665, 15666, 15666, 15671, 15671, 15675, 15676, 15676, 15676, 15680, 15681, 15681, 15682, 15685, 15685, 15685, 15686, 15686, 15687, 15687, 15688, 15688, 15689, 15696, 15696, 15696, 15698, 15699, 15699, 15701, 15702, 15702, 15702, 15703, 15706, 15707, 15708, 15708, 15710, 15710, 15713, 15714, 15714, 15716, 15718, 15718, 15722, 15722, 15723, 15726, 15726, 15726, 15727, 15727, 15727, 15732, 15735, 15737, 15739, 15739, 15742, 15743, 15748, 15751, 15751, 15751, 15752, 15754, 15755, 15763, 15768, 15770, 15770, 15773, 15776, 15777, 15781, 15781, 15786, 15786, 15786, 15786, 15786, 15787, 15788, 15792, 15793, 15794, 15798, 15799, 15799, 15799, 15799, 15799, 15799, 15805, 15807, 15807, 15808, 15808, 15810, 15810, 15812, 15813, 15819, 15822, 15823, 15827, 15830, 15832, 15834, 15837, 15837, 15837, 15837, 15837, 15837, 15838, 15838, 15838, 15838, 15840, 15840, 15848, 15848, 15849, 15850, 15850, 15851, 15853, 15853, 15854, 15857, 15858, 15858, 15860, 15860, 15864, 15865, 15866, 15868, 15868, 15869, 15870, 15870, 15872, 15874, 15877, 15877, 15882, 15886, 15886, 15889, 15889, 15892, 15892, 15892, 15893, 15893, 15893, 15893, 15893, 15895, 15896, 15897, 15898, 15899, 15899, 15899, 15899, 15899, 15900, 15900, 15901, 15904, 15904, 15907, 15910, 15911, 15912, 15912, 15912, 15912, 15912, 15912, 15914, 15915, 15916, 15916, 15917, 15917, 15918, 15920, 15920, 15921, 15921, 15922, 15931, 15936, 15941, 15942, 15943, 15944, 15946, 15946, 15948, 15951, 15957, 15958, 15960, 15962, 15965, 15967, 15967, 15969, 15971, 15971, 15971, 15971, 15972, 15972, 15973, 15973, 15974, 15974, 15975, 15976, 15976, 15976, 15978, 15978, 15978, 15980, 15981, 15982, 15985, 15986, 15987, 15989, 15989, 15989, 15990, 15990, 15991, 15991, 15993, 15994, 15994, 15995, 15995, 15995, 15995, 15995, 15997, 15997, 16000, 16000, 16002, 16002, 16003, 16003, 16003, 16006, 16006, 16006, 16007, 16010, 16015, 16017, 16017, 16017, 16019, 16021, 16022, 16023, 16023, 16025, 16025, 16027, 16028, 16029, 16030, 16032, 16032, 16032, 16032, 16032, 16032, 16032, 16033, 16034, 16035, 16038, 16038, 16039, 16039, 16040, 16041, 16041, 16042, 16042, 16045, 16046, 16046, 16046, 16047, 16047, 16047, 16047, 16047, 16048, 16051, 16052, 16053, 16058, 16058, 16060, 16062, 16064, 16071, 16072, 16072, 16073, 16075, 16075, 16077, 16080, 16080, 16082, 16083, 16083, 16083, 16083, 16084, 16085, 16089, 16089, 16089, 16089, 16092, 16094, 16095, 16096, 16097, 16099, 16099, 16101, 16101]


def get_in_lex_sentences(out_lex_idxs):
    out_lex_sentence_idxs = list(set(out_lex_idxs))
    idxs = [i for i in range(0, 16102)]
    in_lex_sentence_idxs = [x for x in idxs if x not in out_lex_idxs]
    # TO DO: use ID_sentence_idxs to get ID sentences


def get_homograph_dist(WHD_df):
    with open(WHD_df, "rb") as df:
        WHD = pickle.load(df)

    homographs = WHD['homograph'].tolist()
    wordids = WHD['wordid'].tolist()

    homograph_count = defaultdict(lambda: 0)
    for homograph in wordids:
        homograph_count[homograph] += 1

    homographs = list(set(homographs))
    print(homographs)
    print("no. of homographs: ", len(homographs))
    print(homograph_count)






