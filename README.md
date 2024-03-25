# Place questions and human-generated answers: A data analysis approach

## Abstract

This paper investigates place-related questions submitted to search systems and their human-generated answers. Place-based search is motivated by the need to identify places matching some criteria, to identify them in space or relative to other places, or to characterize the qualities of such places. Human place-related questions have thus far been insufficiently studied and differ strongly from typical keyword queries. They thus challenge today's search engines providing only rudimentary geographic information retrieval support. We undertake an analysis of the patterns in place-based questions using a large-scale dataset of questions/answers, MS MARCO V2.1. The results of this study reveal patterns that can inform the design of conversational search systems and in-situ assistance systems, such as autonomous vehicles.

## Dataset
[MSMARCO V2](http://www.msmarco.org/dataset.aspx) is used in this research.

## Overview

Two versions of implement are provided:

### Simple end to end encoding (Reconmend)

#### Requirement

[nltk](https://www.nltk.org)

#### Running Code

```
python simple_end_to_end.py
```

### Complex encoding (Time consuming)

#### Requirement

[Standford parser](https://nlp.stanford.edu/software/lex-parser.shtml): You need to download can configure it.

[EMLo](https://allennlp.org/elmo)

[sklearn](https://scikit-learn.org/stable/)

#### Running Code

Run standford parser, parse the questions and answers, output 10 columns conll format.

```
python parse
```

Encoding

```
python complex_encoding.py
```

### Clustering

Two kinds of representations are used separately to do cluster: semantic encoding and contextual semantic embedding. To do contextual semantic embedding cluster, you need first generate contextual embedding using ```./generate_elmo_embedding.sh```.


### Data

code: encode results

common_word: most commonly used words

gazetteer: gazetteer and abbreviations

place_type: place types

prep: prepositions

verb: action and stative verbs

raw_data: raw dataset

result: results
