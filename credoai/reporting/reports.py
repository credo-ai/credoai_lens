import nbformat as nbf 
import os
import pickle
import textwrap
from nbclient import NotebookClient

class NotebookReport():
    def __init__(self):
        self.nb = nbf.v4.new_notebook()
        self.nb_file_loc = ''
        self.cells = []
    
    def add_cells(self, cells):
        cells = self._preprocess_cell_content(cells)
        self.cells += cells
        for content, cell_type in cells:
            if cell_type == 'markdown':
                self.nb['cells'].append(nbf.v4.new_markdown_cell(content))
            elif cell_type == 'code':
                self.nb['cells'].append(nbf.v4.new_code_cell(content))
    
    def write_notebook(self, file_loc, run=False, **run_kwargs):
        if run:
            self.run_notebook(**run_kwargs)
        nbf.write(self.nb, file_loc)
        self.nb_file_loc = file_loc

    def run_notebook(self):
        client = NotebookClient(self.nb, timeout=600, 
                    kernel_name='python3')

        client.execute()

    def _preprocess_cell_content(self, cells):
        return [(textwrap.dedent(content), cell_type) 
                for content, cell_type in cells]

class AssessmentReport(NotebookReport):
    def __init__(self):
        super().__init__()
        # set up reporter
        load_code="""\
        import pickle
        reporter = pickle.load(open('tmp.pkl','rb'))
        """
        self.add_cells([(load_code, 'code')])
    
    def run_notebook(self, reporter):
        pickle.dump(reporter, open('tmp.pkl', 'wb'))
        client = NotebookClient(self.nb, timeout=600, 
                    kernel_name='python3')

        client.execute()
        os.remove('tmp.pkl')

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
        {toc}
        ## Basic Information
        
        Model Information
            
        * Model: {names['model']}
        * Dataset: {names['dataset']}
        """
        cells = [(boiler_plate, 'markdown')]
        self.add_cells(cells)
    
    def get_toc(self):
        toc = """**Table of Contents**

        1. [Basic Information](#Basic-Information)
        1. [Executive Summary](#Executive-Summary)
        1. [Technical Reports](#Technical-Reports)
        """
        for reporter in self.reporters:
            tmp = f"""\
                1. [{reporter.assessment.name} Report](#{reporter.assessment.name}-Report)
            """
            toc += textwrap.indent(textwrap.dedent(tmp), '    ')
        return toc

    def create_report(self, lens, directory):
        self.create_boiler_plate(lens)
        cells = [
            ("""## Executive Summary
            
            We have executive stuff here
            """, 'markdown'),
            ("""## Technical Reports""", 'markdown')
        ]
        self.add_cells(cells)
        loc = os.path.join(directory, 'main_report.ipynb')
        self.write_notebook(loc, run=True)
        self.add_technical()

    def add_technical(self):
        for reporter in self.reporters:
            technical_notebook = reporter.report
            # Reading the notebooks
            first_notebook = nbf.read(self.nb_file_loc, 4)
            second_notebook = nbf.read(technical_notebook.nb_file_loc, 4)

            # Creating a new notebook
            final_notebook = nbf.v4.new_notebook(metadata=first_notebook.metadata)

            # Concatenating the notebooks
            final_notebook.cells = first_notebook.cells + second_notebook.cells

            self.nb = final_notebook
            # Saving the new notebook 
            self.write_notebook('main_report.ipynb')
