# Creating a Sample Document

To fully test the AI Agent Workflow, you'll need a sample PDF document. Here's how to create one:

## Option 1: Use a Sample Business Document

Download a sample business document PDF (20-25 pages) from websites that offer free sample business plans, annual reports, or whitepapers.

Suggested sources:
- [Sample-Documents.com](https://sample-documents.com/business-documents.html)
- [PDF Templates on Template.net](https://www.template.net/business/pdf-templates/)
- [SEC EDGAR Database](https://www.sec.gov/edgar/searchedgar/companysearch.html) (for public company annual reports)

## Option 2: Create a Sample Document

If you prefer to create your own sample document:

1. Open a word processor (e.g., Microsoft Word, Google Docs)
2. Create a 20-25 page document with:
   - Executive Summary
   - Strategic Goals section
   - Competitive Landscape analysis
   - Financial Projections
   - Risk Assessment
   - Tables of data and some charts/figures
3. Export/save as PDF

## Option 3: Use the Included Sample Generator

We've included a Python script to generate a sample business document:

```bash
# Install required packages
pip install faker python-docx matplotlib

# Run the generator script
python utils/sample_doc_generator.py

# This will create a sample PDF in the samples/ directory
```

## Testing with the Sample Document

After obtaining a sample PDF, use it to test the system:

```bash
# Run the system with the sample document
python main.py --document path/to/sample.pdf --interactive
```
