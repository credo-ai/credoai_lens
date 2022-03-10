import nbformat as nbf 
import os
import cloudpickle
import textwrap
from datetime import datetime
from inspect import cleandoc
from nbclient import NotebookClient
from traitlets.config import Config
import asyncio
import nbconvert
import nest_asyncio

HTML_CONFIG = Config()
HTML_CONFIG.TemplateExporter.exclude_input = True
HTML_CONFIG.TemplateExporter.exclude_input_prompt = True
HTML_CONFIG.TemplateExporter.exclude_output_prompt = True

class NotebookReport():
    def __init__(self):
        self.nb = nbf.v4.new_notebook()
        self.cells = []
        self.add_cells([self.get_style_cell()])
    
    def add_cells(self, cells):
        cells = self._preprocess_cell_content(cells)
        self.cells += cells
        for content, cell_type in cells:
            if cell_type == 'markdown':
                self.nb['cells'].append(nbf.v4.new_markdown_cell(content))
            elif cell_type == 'code':
                self.nb['cells'].append(nbf.v4.new_code_cell(content))
        
    def write_notebook(self, file_loc):
        """Write notebook to file

        Parameters
        ----------
        file_loc : str
            file location to save notebook
            
        Returns
        -------
        self
        """        
        if file_loc.endswith('.html'):
            html = self.to_html()
            with open(file_loc, 'w') as f:
                f.write(html)
        else:
            nbf.write(self.nb, file_loc)
        return self

    def get_style_cell(self):
        cell = ("""\
        %%html
        <style>
        ::marker {
            unicode-bidi: isolate;
            font-variant-numeric: tabular-nums;
            text-transform: none;
            text-indent: 0px !important;
            text-align: start !important;
            text-align-last: start !important;
        }
        </style>
        """, "code"
        )
        return cell
        
    def run_notebook(self):
        client = NotebookClient(self.nb, timeout=600, 
                    kernel_name='python3')
        loop = asyncio.new_event_loop()
        nest_asyncio.apply(loop)
        loop.run_until_complete(client.async_execute())
        return self

    def to_html(self):
        """Converts notebook to html"""
        html_exporter = nbconvert.HTMLExporter(config=HTML_CONFIG)
        (body, resources) = html_exporter.from_notebook_node(self.nb)
        return body
    
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
        load_code="import cloudpickle\n"
        for key, val in self.needed_artifacts.items():
            load_code += f"{key} = cloudpickle.load(open('{key}.pkl','rb'))\n"
        load_code += "%config InlineBackend.figure_formats = ['svg', 'png']"
        self.add_cells([(load_code, 'code')])
    
    def run_notebook(self):
        pickle_files = []
        for key, val in self.needed_artifacts.items():
            pickle_files.append(f'{key}.pkl')
            cloudpickle.dump(val, open(pickle_files[-1], 'wb'))
        client = NotebookClient(self.nb, timeout=600, 
                    kernel_name='python3')
        loop = asyncio.new_event_loop()
        nest_asyncio.apply(loop)
        loop.run_until_complete(client.async_execute())
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
        # <span style="color:#3b07b4;font-weight:bold">{self.name}</span>
        
        **Table of Contents**\n
"""\
        f"{textwrap.indent(toc, ' '*8)}"\
        f"""
        
        **Basic Information**
        
        * Creation time: {datetime.now().strftime("%Y-%m-%d %H:%M")}
        * Model: {names['model']}
        * Dataset: {names['dataset']}
        """
        html_code = """\
        from IPython.display import HTML

        HTML('''
        <script>
        window.addEventListener('load', function() {
	    let message = { height: document.body.scrollHeight, width: document.body.scrollWidth };	
	    window.top.postMessage(message, "*");
        });
        </script>

        <script>
        $('div.input').hide();
        </script>
        ''')
        """
        cells = [(boiler_plate, 'markdown'),
                 (html_code, 'code')
                 ]
        self.add_cells(cells)
    
    def get_toc(self):
        toc = """1. [Basic Information](#Basic-Information)
        1. [Technical Reports](#Technical-Reports)
        """
        toc = cleandoc(toc)
        for reporter in self.reporters:
            tmp = f"\n1. [{reporter.assessment.name} Report](#{reporter.assessment.name}-Report)"
            toc += textwrap.indent(tmp, '    ')
            result_link  = f"\n1. [{reporter.assessment.name} Results](#{reporter.assessment.name}-Results)"
            toc += textwrap.indent(result_link, '        ')
            result_table_link  = f"\n1. [{reporter.assessment.name} Result Tables](#{reporter.assessment.name}-Result-Tables)"
            toc += textwrap.indent(result_table_link, '        ')
        return toc

    def create_report(self, lens):
        self.create_boiler_plate(lens)
        cells = [
            ("""# <span style="color:#3b07b4">Technical Reports</span>""", 'markdown')
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
