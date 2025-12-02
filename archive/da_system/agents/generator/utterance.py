class Utterance(object):
    def __init__(self, raw_text=None, tokens=None, logical_form=None, template=None, ambiguous_template=False, agent=None):
        self.text = raw_text
        self.tokens = tokens
        self.lf = logical_form
        self.template = template
        self.ambiguous_template = ambiguous_template
        self.agent = agent

    def to_dict(self):
        return {
                'logical_form': self.lf.to_dict(),
                'template': self.template,
                }

    def __str__(self):
        s = []
        s.append('-'*20)
        if self.text:
            s.append(self.text)
        if self.lf:
            s.append(self.lf)
        if self.template:
            s.append(' '.join(self.template))
        return '\n'.join([str(x) for x in s])

class LogicalForm(object):
    def __init__(self, intent, **kwargs):
        self.intent = intent
        for k, v in kwargs.items():
            setattr(self, k, v) # LogicalFormオブジェクトにprice属性を追加

    def to_dict(self):
        attrs = vars(self)
        attrs['intent'] = self.intent
        return attrs

    def __str__(self):
        attrs = vars(self)
        s = ' '.join(['{}={}'.format(k, v) for k, v in attrs.items()])
        return s