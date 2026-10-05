"""Two-domain comparison and data-grounded answers for the detailed brief."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.domain_compare import compare_domains, domain_answer, domain_profile
from src.hw2_answers import part5_answers, part6_answer, part7_answers, part8_answer
from src.parse_pmc import Document
from src.zipf_experiment import AbstractCorpus, analyze_corpus


def corpus(prefix, texts):
    docs = [Document(f'PMID{prefix}{i}', f'{prefix}{i}', '', 'Title', text, '', '', '2026') for i, text in enumerate(texts, 1)]
    return AbstractCorpus(docs, prefix, prefix + '-fingerprint', {})


MEDICAL = corpus('1', ['Insulin lowers glucose in patients. Patients received insulin.', 'Glucose and insulin in diabetes patients.',
                       'The trial measured glucose. Diabetes patients improved.'])
COMPUTING = corpus('2', ['The network model learned. The model used layers.', 'An algorithm trained the network model.',
                         'Learning improved the model. The network learned features.'])


class DomainTests(unittest.TestCase):
    def test_profile_counts_only_the_first_documents(self):
        profile = domain_profile(MEDICAL, 2, 'C')
        self.assertEqual(profile['documents'], 2)
        self.assertEqual(profile['top_terms'][0], 'insulin')
        self.assertEqual(profile['tokens'], sum(r['CF'] for r in profile['rows']))
        with self.assertRaises(ValueError):
            domain_profile(MEDICAL, 4, 'C')

    def test_comparison_overlap_and_answer(self):
        result = compare_domains(MEDICAL, COMPUTING, condition='C')
        self.assertEqual(result['size'], 3)
        self.assertEqual(result['shared_vocabulary'], 1)  # 'improved' occurs in both domains
        self.assertIn('model', result['distinctive_B'])
        text = domain_answer(result, ('Medical', 'Computing'))
        self.assertIn('Medical', text)
        self.assertIn(f"{result['A']['zipf_exponent']:.3f}", text)


class AnswerTests(unittest.TestCase):
    def test_answers_cover_every_required_question_with_measured_numbers(self):
        result = analyze_corpus(MEDICAL)
        self.assertEqual(len(part5_answers(result)), 4)
        self.assertEqual(len(part6_answer(result)), 4)
        self.assertEqual(len(part7_answers(result)), 3)
        self.assertEqual(len(part8_answer(result)), 2)
        vocabulary = result['conditions']['A']['stats']['vocabulary']
        self.assertIn(f'A: {vocabulary:,}', part6_answer(result)['1 · Vocabulary size'])


if __name__ == '__main__':
    unittest.main()
