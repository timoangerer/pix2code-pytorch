import json
from .node import Node
from .utils import Utils
from pathlib import Path

class HTMLTranspiler:
    def __init__(self, dsl_mapping_file_path=Path(Path(__file__).parent, "web-dsl-mapping.json")):
        with open(dsl_mapping_file_path) as data_file:
            self.dsl_mapping = json.load(data_file)

        self.opening_tag = self.dsl_mapping["opening-tag"]
        self.closing_tag = self.dsl_mapping["closing-tag"]
        self.content_holder = self.opening_tag + self.closing_tag

        self.root = Node("body", None, self.content_holder)

    def transpile(self, tokens, output_file_path=None, insert_random_text=False):
        if not tokens or len(tokens) <= 0:
            raise ValueError(f'Tokens must be a none-empty array')

        rendering_function = None
        if insert_random_text:
            rendering_function = Utils.render_content_with_text

        current_parent = self.root
        last_inserted_element = None

        for token in tokens:
            if token == self.opening_tag:
                current_parent = last_inserted_element
            elif token == self.closing_tag:
                current_parent = current_parent.parent
            else:
                element = Node(token, current_parent, self.content_holder)
                current_parent.add_child(element)
                last_inserted_element = element

        output_html = self.root.render(self.dsl_mapping, rendering_function)

        if output_file_path:
            with open(output_file_path, 'w') as output_file:
                output_file.write(output_html)
        
        return output_html