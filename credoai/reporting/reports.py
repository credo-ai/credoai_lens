import nbformat as nbf 
import os
import pickle
import textwrap
from inspect import cleandoc
from nbclient import NotebookClient
from nbconvert import HTMLExporter

class NotebookReport():
    def __init__(self):
        self.nb = nbf.v4.new_notebook()
        self.cells = []
    
    def add_cells(self, cells):
        cells = self._preprocess_cell_content(cells)
        self.cells += cells
        for content, cell_type in cells:
            if cell_type == 'markdown':
                self.nb['cells'].append(nbf.v4.new_markdown_cell(content))
            elif cell_type == 'code':
                self.nb['cells'].append(nbf.v4.new_code_cell(content))
    
    def export_notebook(self, file_loc, run=False):
        """Writes notebook to html"""
        if run:
            self.run_notebook()
        html_exporter = HTMLExporter(template_name = 'classic')
        (body, resources) = html_exporter.from_notebook_node(self.nb)
        with open(file_loc, "w") as html_nb:
            # Writing data to a file
            html_nb.write(body)
        return self

    def write_notebook(self, file_loc, run=False):
        """Writes notebook to file"""
        if run:
            self.run_notebook()
        nbf.write(self.nb, file_loc)
        return self

    def run_notebook(self):
        client = NotebookClient(self.nb, timeout=600, 
                    kernel_name='python3')

        client.execute()
        return self

    def _preprocess_cell_content(self, cells):
        return [(cleandoc(content), cell_type) 
                for content, cell_type in cells]

class AssessmentReport(NotebookReport):
    def __init__(self, needed_artifacts=None):
        """Assessment version of the report
        
        Parameters
        ---------
        needed_artifacts : dict
            dictionary of artifacts that will be pickled to run the report
        """
        super().__init__()
        self.needed_artifacts = needed_artifacts
        # set up reporter
        load_code="import pickle\n"
        for key, val in self.needed_artifacts.items():
            load_code += f"{key} = pickle.load(open('{key}.pkl','rb'))"
        self.add_cells([(load_code, 'code')])
    
    def run_notebook(self):
        pickle_files = []
        for key, val in self.needed_artifacts.items():
            pickle_files.append(f'{key}.pkl')
            pickle.dump(val, open(pickle_files[-1], 'wb'))
        client = NotebookClient(self.nb, timeout=600, 
                    kernel_name='python3')

        client.execute()
        for f in pickle_files:
            os.remove(f)
        return self

class MainReport(NotebookReport):
    def __init__(self, report_name, reporters):
        super().__init__()
        self.name = report_name
        self.reporters = reporters

    def create_boiler_plate(self, lens):
        names = lens.get_artifact_names()
        toc = self.get_toc()
        boiler_plate=f"""\
        # {self.name}
        
        **Table of Contents**\n
"""\
        f"{textwrap.indent(toc, ' '*8)}"\
        f"""
        
        **Basic Information**
        
        Model Information
            
        * Model: {names['model']}
        * Dataset: {names['dataset']}
        """
        cells = [(boiler_plate, 'markdown')]
        self.add_cells(cells)
    
    def get_toc(self):
        toc = """1. [Basic Information](#Basic-Information)
        1. [Executive Summary](#Executive-Summary)
        1. [Technical Reports](#Technical-Reports)
        """
        toc = cleandoc(toc)
        for reporter in self.reporters:
            tmp = f"\n1. [{reporter.assessment.name} Report](#{reporter.assessment.name}-Report)"
            toc += textwrap.indent(tmp, '    ')
        return toc

    def create_report(self, lens):
        self.create_boiler_plate(lens)
        cells = [
            ("""## Executive Summary
            
            We have executive stuff here
            """, 'markdown'),
            ("""# Technical Reports""", 'markdown')
        ]
        self.add_cells(cells)
        self.run_notebook()
        self.add_technical()
        return self

    def add_technical(self, run_technical=True):
        for reporter in self.reporters:
            technical_notebook = reporter.report
            # Reading the notebooks
            first_notebook = self.nb
            if run_technical:
                second_notebook = technical_notebook.run_notebook().nb
            else:
                second_notebook = technical_notebook.nb
            # Creating a new notebook
            cat_notebook = nbf.v4.new_notebook(metadata=first_notebook.metadata)
            # Concatenating the notebooks
            cat_notebook.cells = first_notebook.cells + second_notebook.cells
            self.nb = cat_notebook
        return self
