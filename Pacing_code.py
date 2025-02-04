import os
import logging
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return False
    return True


def main():
    setup_logging()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_name = 'Weekly_Pacing_Test.csv'
    output_file_name = 'Pacing_Document.xlsx'
    input_file_path = os.path.join(base_dir, input_file_name)
    output_file_path = os.path.join(base_dir, output_file_name)

    if not os.path.isfile(input_file_path):
        logging.error(f"Input file not found: {input_file_path}")
        return

    try:
        df = pd.read_csv(input_file_path)
        logging.info("Data successfully read from CSV")
    except Exception as e:
        logging.error(f"Error reading the CSV file: {e}")
        return

    required_columns = ['Budget Segment Start Date', 'Budget Segment End Date', 'Date',
                        'Budget Segment Budget', 'Total Media Cost (Advertiser Currency)', 'Insertion Order']
    if not validate_columns(df, required_columns):
        return

    date_columns = ['Budget Segment Start Date', 'Budget Segment End Date', 'Date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    df['Budget Segment Budget'] = pd.to_numeric(
        df['Budget Segment Budget'].astype(str).str.replace(r'[$,\s]', '', regex=True), errors='coerce')
    df['Total Media Cost (Advertiser Currency)'] = pd.to_numeric(
        df['Total Media Cost (Advertiser Currency)'].astype(str).str.replace(r'[$,\s]', '', regex=True),
        errors='coerce')

    df.dropna(subset=['Date', 'Total Media Cost (Advertiser Currency)', 'Budget Segment Budget'], inplace=True)

    pacing_df = df.groupby(['Insertion Order']).agg({
        'Total Media Cost (Advertiser Currency)': 'sum',
        'Budget Segment Budget': 'first',
        'Budget Segment Start Date': 'first',
        'Budget Segment End Date': 'first'
    }).reset_index()

    today = pd.to_datetime('today')
    pacing_df['Days Elapsed'] = (today - pacing_df['Budget Segment Start Date']).dt.days.clip(lower=0)
    pacing_df['Total Days'] = (
                pacing_df['Budget Segment End Date'] - pacing_df['Budget Segment Start Date']).dt.days.clip(lower=1)

    pacing_df['Time Elapsed (%)'] = (pacing_df['Days Elapsed'] / pacing_df['Total Days']).fillna(0) * 100
    pacing_df['Remaining Budget'] = (
                pacing_df['Budget Segment Budget'] - pacing_df['Total Media Cost (Advertiser Currency)']).fillna(0)
    pacing_df['Daily Required Spend'] = (
                pacing_df['Remaining Budget'] / (pacing_df['Total Days'] - pacing_df['Days Elapsed']).replace(0,
                                                                                                              1)).fillna(
        0)
    pacing_df['Average Daily Spend'] = (
                pacing_df['Total Media Cost (Advertiser Currency)'] / pacing_df['Days Elapsed'].replace(0, 1)).fillna(0)
    pacing_df['Estimated Spend'] = pacing_df['Average Daily Spend'] * pacing_df['Total Days']
    pacing_df['Pacing Rate (%)'] = (pacing_df['Total Media Cost (Advertiser Currency)'] / pacing_df[
        'Budget Segment Budget']).fillna(0) * 100

    summary_data = {
        'Total Budget': [pacing_df['Budget Segment Budget'].sum()],
        'Total Spend': [pacing_df['Total Media Cost (Advertiser Currency)'].sum()],
        'Overall Pacing Rate (%)': [pacing_df['Pacing Rate (%)'].mean()]
    }
    summary_df = pd.DataFrame(summary_data)

    key_insights = [
        f"Total budget allocated: {summary_df['Total Budget'][0]:,.2f}",
        f"Total spend to date: {summary_df['Total Spend'][0]:,.2f}, Average pacing rate: {summary_df['Overall Pacing Rate (%)'][0]:.2f}%",
        "Monitor IOs with pacing rate exceeding 100%"
    ]

    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        pacing_df.to_excel(writer, index=False, sheet_name='Pacing Summary')
        summary_df.to_excel(writer, index=False, sheet_name='Summary')

        workbook = writer.book
        summary_sheet = writer.sheets['Summary']

        for idx, insight in enumerate(key_insights, start=summary_df.shape[0] + 2):
            summary_sheet.cell(row=idx, column=1).value = insight

        chart = BarChart()
        chart.title = "Budget vs Total Spend"
        chart.x_axis.title = 'Metric'
        chart.y_axis.title = 'Value'
        data = Reference(summary_sheet, min_col=1, min_row=1, max_col=2, max_row=2)
        categories = Reference(summary_sheet, min_col=1, min_row=2, max_row=2)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(categories)
        summary_sheet.add_chart(chart, "E5")

    logging.info(f"Pacing document created at {output_file_path}")


if __name__ == "__main__":
    main()
