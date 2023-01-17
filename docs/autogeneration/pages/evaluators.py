from docs.autogeneration.formatter import (
    create_page_area,
    create_title,
    extract_docstring_info_from_evaluator,
)
from credoai.lens.pipeline_creator import build_list_of_evaluators


def create_all_evaluator_pages():
    all_ev = build_list_of_evaluators()
    for evaluator in all_ev:
        try:
            doc = extract_docstring_info_from_evaluator(evaluator)
            page_name = evaluator.name.lower()
            if "(Experimental)" in doc:
                with open(
                    f"./pages/evaluators/experimental/{page_name}.rst", "w"
                ) as text_file:
                    text_file.write(doc)
            else:
                with open(f"./pages/evaluators/{page_name}.rst", "w") as text_file:
                    text_file.write(doc)
        except:
            print(f"{evaluator.name} docstring not found")


if __name__ == "__main__":
    create_all_evaluator_pages()
