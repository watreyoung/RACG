import json


def add_lang_by_task(target_str, task, sub_task):
    if task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'conala' or task == 'hearthstone':
        target_str = '<python> ' + target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source

    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_conala_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples

def read_hearthstone_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


