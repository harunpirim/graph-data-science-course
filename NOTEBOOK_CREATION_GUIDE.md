# Step-by-Step Guide: Creating and Pushing New Notebooks

This guide provides detailed instructions for creating new Jupyter notebooks and pushing them to your GitHub repository with working Google Colab links.

## 🚀 Quick Start (Automated Method)

### Step 1: Generate Notebooks Using the Script

```bash
# Run the automated script to create notebooks
python create_all_notebooks.py
```

This will create:
- `02-connectivity.ipynb`
- `03-graph-machine-learning.ipynb`

### Step 2: Add and Push to GitHub

```bash
# Add all new notebooks
git add notebooks/*.ipynb

# Commit the changes
git commit -m "Add connectivity and GML notebooks with Google Colab compatibility"

# Push to GitHub
git push
```

### Step 3: Test Google Colab Links

Your Google Colab links will now work:
- https://colab.research.google.com/github/harunpirim/graph-data-science-course/blob/main/notebooks/02-connectivity.ipynb
- https://colab.research.google.com/github/harunpirim/graph-data-science-course/blob/main/notebooks/03-graph-machine-learning.ipynb

---

## 📝 Manual Method (Step-by-Step)

### Method 1: Using Python Scripts

#### Step 1: Create a Notebook Generation Script

Create a file `create_single_notebook.py`:

```python
import json

def create_notebook(filename, title, cells):
    """Create a Jupyter notebook with the given cells"""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(f'notebooks/{filename}', 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Created {filename} successfully!")

# Define your notebook cells
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Your Notebook Title\n",
            "\n",
            "Description of your notebook content."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Your code here\n",
            "import networkx as nx\n",
            "import matplotlib.pyplot as plt\n",
            "print('Hello, Graph Data Science!')"
        ]
    }
]

# Create the notebook
create_notebook("your-notebook-name.ipynb", "Your Notebook Title", cells)
```

#### Step 2: Run the Script

```bash
python create_single_notebook.py
```

#### Step 3: Add to Git and Push

```bash
# Add the new notebook
git add notebooks/your-notebook-name.ipynb

# Commit
git commit -m "Add your-notebook-name notebook"

# Push
git push
```

### Method 2: Using Jupyter Notebook Directly

#### Step 1: Create Notebook in Jupyter

```bash
# Start Jupyter
jupyter notebook
```

#### Step 2: Create New Notebook

1. Click "New" → "Python 3"
2. Add your content (markdown and code cells)
3. Save as `notebooks/your-notebook-name.ipynb`

#### Step 3: Add to Git and Push

```bash
# Add the notebook
git add notebooks/your-notebook-name.ipynb

# Commit
git commit -m "Add your-notebook-name notebook"

# Push
git push
```

### Method 3: Using Google Colab

#### Step 1: Create in Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Add your content

#### Step 2: Download and Add to Repository

1. File → Download → Download .ipynb
2. Move the file to `notebooks/your-notebook-name.ipynb`
3. Add to Git and push

---

## 🔧 Notebook Structure Template

Here's a template for creating well-structured notebooks:

```python
# Template for creating a new notebook
notebook_cells = [
    # Title and Introduction
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Your Topic Title\n",
            "\n",
            "This notebook demonstrates [topic] using [libraries].\n",
            "\n",
            "## Learning Objectives\n",
            "- Objective 1\n",
            "- Objective 2\n",
            "- Objective 3"
        ]
    },
    
    # Import Libraries
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import required libraries\n",
            "import networkx as nx\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "\n",
            "# Set up plotting\n",
            "%matplotlib inline\n",
            "plt.rcParams['figure.figsize'] = (12, 8)"
        ]
    },
    
    # Section 1
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Section Title\n",
            "\n",
            "Description of this section."
        ]
    },
    
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Your code for section 1\n",
            "print('Section 1 code')"
        ]
    },
    
    # Section 2
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Section Title\n",
            "\n",
            "Description of this section."
        ]
    },
    
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Your code for section 2\n",
            "print('Section 2 code')"
        ]
    },
    
    # Summary
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Summary and Key Insights\n",
            "\n",
            "### Key Takeaways:\n",
            "1. Takeaway 1\n",
            "2. Takeaway 2\n",
            "3. Takeaway 3\n",
            "\n",
            "### Applications:\n",
            "- Application 1\n",
            "- Application 2\n",
            "- Application 3"
        ]
    }
]
```

---

## 📋 Checklist for New Notebooks

### Before Creating:
- [ ] Decide on the topic and learning objectives
- [ ] Plan the structure (sections, examples, visualizations)
- [ ] Choose appropriate datasets or examples

### During Creation:
- [ ] Use clear, descriptive markdown cells
- [ ] Include code comments and explanations
- [ ] Add visualizations where appropriate
- [ ] Test code cells to ensure they work
- [ ] Include real-world examples

### Before Pushing:
- [ ] Test the notebook locally
- [ ] Ensure all imports are included
- [ ] Check that visualizations render correctly
- [ ] Verify the notebook structure is correct

### After Pushing:
- [ ] Test the Google Colab link
- [ ] Update the README.md if needed
- [ ] Share with students/colleagues

---

## 🛠️ Troubleshooting

### Common Issues:

#### 1. Notebook Won't Open in Google Colab
**Solution**: Ensure the notebook has the correct JSON structure and is properly committed to GitHub.

#### 2. Import Errors in Google Colab
**Solution**: Add installation commands at the beginning of your notebook:
```python
# Install required packages
!pip install networkx matplotlib numpy pandas seaborn
```

#### 3. Large File Size
**Solution**: Remove output cells and large data before committing:
```bash
# Clean notebook (remove outputs)
jupyter nbconvert --clear-output --inplace notebooks/your-notebook.ipynb
```

#### 4. Git Issues
**Solution**: Check your git status and resolve conflicts:
```bash
git status
git add .
git commit -m "Your commit message"
git push
```

---

## 🎯 Best Practices

### Content:
- Start with clear learning objectives
- Use progressive complexity (simple to advanced)
- Include real-world examples
- Provide clear explanations for each concept
- End with a summary and key insights

### Code:
- Use consistent naming conventions
- Add comments to explain complex code
- Include error handling where appropriate
- Test all code cells before pushing

### Documentation:
- Use descriptive markdown cells
- Include references and citations
- Provide links to additional resources
- Add contact information for questions

### Version Control:
- Use descriptive commit messages
- Commit frequently with small changes
- Test notebooks after each major change
- Keep backup copies of important notebooks

---

## 📚 Example: Creating a Complete Notebook

Here's a complete example of creating a new notebook:

### Step 1: Create the Script

```python
# create_visualization_notebook.py
import json

def create_visualization_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Graph Visualization Techniques\n",
                "\n",
                "This notebook demonstrates various graph visualization techniques using NetworkX and Matplotlib.\n",
                "\n",
                "## Learning Objectives\n",
                "- Understand different layout algorithms\n",
                "- Create effective visualizations\n",
                "- Customize graph appearance\n",
                "- Analyze visual patterns"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import networkx as nx\n",
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "\n",
                "%matplotlib inline\n",
                "plt.rcParams['figure.figsize'] = (12, 8)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Basic Graph Visualization\n",
                "\n",
                "Let's start with basic graph visualization techniques."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create a sample graph\n",
                "G = nx.karate_club_graph()\n",
                "print(f\"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\")\n",
                "\n",
                "# Basic visualization\n",
                "plt.figure(figsize=(10, 8))\n",
                "nx.draw(G, with_labels=True, node_color='lightblue', \n",
                "        node_size=500, font_size=10)\n",
                "plt.title('Karate Club Network')\n",
                "plt.show()"
            ]
        }
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('notebooks/04-graph-visualization.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print("Created 04-graph-visualization.ipynb successfully!")

if __name__ == "__main__":
    create_visualization_notebook()
```

### Step 2: Run and Push

```bash
# Run the script
python create_visualization_notebook.py

# Add to git
git add notebooks/04-graph-visualization.ipynb

# Commit
git commit -m "Add graph visualization notebook"

# Push
git push
```

### Step 3: Test

Visit: https://colab.research.google.com/github/harunpirim/graph-data-science-course/blob/main/notebooks/04-graph-visualization.ipynb

---

## 🎉 Success!

You now have a complete workflow for creating and pushing new notebooks to your Graph Data Science course repository. Each notebook will have working Google Colab links and can be used by students worldwide!

Remember to:
- Keep notebooks focused and well-structured
- Test all code before pushing
- Use descriptive commit messages
- Update documentation as needed
- Share your work with the community
