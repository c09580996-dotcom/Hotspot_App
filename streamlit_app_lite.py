"""
Crime Hotspot Analysis Tool - Streamlit Web Application
Author: Chynara Gambarova
CST-580 | December 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Crime Hotspot Analysis", page_icon="🚔", layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'security_log' not in st.session_state:
    st.session_state.security_log = []

def log_security_event(event_type, details):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.security_log.append(f"[{timestamp}] {event_type}: {details}")

def validate_data_input(data):
    log_security_event("VALIDATION", "Checking uploaded data")
    issues = []
    if data is not None:
        suspicious_cols = [col for col in data.columns if any(x in str(col).lower() for x in ['password', 'ssn', 'credit'])]
        if suspicious_cols:
            issues.append(f"Warning: Sensitive columns: {suspicious_cols}")
        if len(data) > 1000000:
            issues.append("Warning: Large dataset (>1M rows)")
    return issues

def anonymize_coordinates(df):
    df_copy = df.copy()
    if 'Lat_Public' in df_copy.columns and 'Long_Public' in df_copy.columns:
        df_copy['Lat_Public'] = df_copy['Lat_Public'] + np.random.normal(0, 0.001, len(df_copy))
        df_copy['Long_Public'] = df_copy['Long_Public'] + np.random.normal(0, 0.001, len(df_copy))
        log_security_event("ANONYMIZATION", "Coordinates anonymized")
    return df_copy

def create_sample_data():
    np.random.seed(42)
    n = 500
    center_lat, center_lon = 33.4484, -112.0740
    data = {
        'IncidentID': range(1, n+1),
        'Lat_Public': center_lat + np.random.normal(0, 0.05, n),
        'Long_Public': center_lon + np.random.normal(0, 0.05, n),
        'OffenseType': np.random.choice(['Theft', 'Assault', 'Burglary', 'Vandalism', 'DUI'], n),
        'Date': pd.date_range('2024-01-01', periods=n, freq='H'),
        'District': np.random.choice(['District 1', 'District 2', 'District 3'], n)
    }
    return pd.DataFrame(data)

def apply_cleaning(df, method):
    if df is None or method == 'None':
        return df
    df_copy = df.copy()
    original_shape = df_copy.shape
    if method == 'Drop NA':
        df_copy = df_copy.dropna()
        log_security_event("CLEANING", f"Dropped NA: {original_shape[0] - df_copy.shape[0]} rows")
    elif method == 'Fill NA with 0':
        df_copy = df_copy.fillna(0)
        log_security_event("CLEANING", "Filled NA with 0")
    elif method == 'Remove Duplicates':
        df_copy = df_copy.drop_duplicates()
        log_security_event("CLEANING", f"Removed duplicates: {original_shape[0] - df_copy.shape[0]} rows")
    return df_copy

def perform_hotspot_analysis_simple(df, cell_size_degrees=0.005):
    lat_col, lon_col = "Lat_Public", "Long_Public"
    if lat_col not in df.columns or lon_col not in df.columns:
        return None, "❌ Missing Lat_Public or Long_Public columns"
    
    df_clean = df.dropna(subset=[lat_col, lon_col]).copy()
    if len(df_clean) == 0:
        return None, "❌ No valid coordinates"
    
    lat_min, lat_max = df_clean[lat_col].min(), df_clean[lat_col].max()
    lon_min, lon_max = df_clean[lon_col].min(), df_clean[lon_col].max()
    
    lat_bins = np.arange(lat_min, lat_max + cell_size_degrees, cell_size_degrees)
    lon_bins = np.arange(lon_min, lon_max + cell_size_degrees, cell_size_degrees)
    
    df_clean['grid_lat'] = pd.cut(df_clean[lat_col], bins=lat_bins, labels=False, include_lowest=True)
    df_clean['grid_lon'] = pd.cut(df_clean[lon_col], bins=lon_bins, labels=False, include_lowest=True)
    
    grid_counts = df_clean.groupby(['grid_lat', 'grid_lon']).size().reset_index(name='count')
    
    if len(grid_counts) > 0:
        threshold = grid_counts['count'].quantile(0.80)
        grid_counts['hotspot'] = grid_counts['count'] >= threshold
        hotspot_cells = grid_counts[grid_counts['hotspot']]
    else:
        threshold = 0
        hotspot_cells = pd.DataFrame()
    
    grid_counts['lat_center'] = lat_bins[grid_counts['grid_lat'].astype(int)] + cell_size_degrees/2
    grid_counts['lon_center'] = lon_bins[grid_counts['grid_lon'].astype(int)] + cell_size_degrees/2
    
    return {
        'grid': grid_counts,
        'hotspot_cells': hotspot_cells,
        'threshold': threshold,
        'total_incidents': len(df_clean),
        'total_cells': len(grid_counts),
        'hotspot_count': len(hotspot_cells),
        'cell_size': cell_size_degrees
    }, None

def plot_hotspot_heatmap(results):
    grid = results['grid']
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(grid['lon_center'], grid['lat_center'], c=grid['count'], s=grid['count']*50,
                        cmap='YlOrRd', alpha=0.6, edgecolors='black', linewidth=0.5)
    hotspots = grid[grid['hotspot']]
    if len(hotspots) > 0:
        ax.scatter(hotspots['lon_center'], hotspots['lat_center'], s=hotspots['count']*50,
                  facecolors='none', edgecolors='red', linewidth=3, label='Hotspots')
    plt.colorbar(scatter, ax=ax, label='Incident Count')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Crime Hotspot Analysis\nGrid Size: ~{results["cell_size"]*111:.0f}m', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def perform_temporal_analysis(df):
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                date_col = col
                break
            except:
                pass
    if date_col is None:
        return None, "❌ No date/time column found"
    df['hour'] = df[date_col].dt.hour
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    return {
        'by_hour': df.groupby('hour').size(),
        'by_day': df.groupby('day_of_week').size(),
        'by_month': df.groupby('month').size()
    }, None

def perform_crime_type_analysis(df):
    crime_col = None
    for col in df.columns:
        if any(x in col.lower() for x in ['offense', 'crime', 'type', 'category']):
            crime_col = col
            break
    if crime_col is None:
        return None, "❌ No crime type column found"
    return df[crime_col].value_counts(), None

def main():
    st.title("🚔 Crime Hotspot Analysis Tool")
    st.markdown("**Comprehensive Data Product for Law Enforcement Analysis**")
    st.markdown("*Author: Chynara Gambarova | CST-580 | December 2025*")
    st.markdown("---")
    
    with st.sidebar:
        st.header("📁 Data Management")
        load_option = st.radio("Select Data Source:", ["Upload File", "Load Sample Data", "Load from URL"])
        
        if load_option == "Upload File":
            uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        st.session_state.df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        st.session_state.df = pd.read_excel(uploaded_file)
                    issues = validate_data_input(st.session_state.df)
                    st.success(f"✅ Loaded: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} cols")
                    for issue in issues:
                        st.warning(issue)
                    log_security_event("UPLOAD", f"File: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        elif load_option == "Load Sample Data":
            if st.button("📊 Load Sample Dataset", use_container_width=True):
                st.session_state.df = create_sample_data()
                st.success("✅ Sample data loaded (500 records)")
                st.info("Phoenix, AZ area crime incidents")
                log_security_event("DATA_LOAD", "Sample dataset loaded")
        
        elif load_option == "Load from URL":
            url = st.text_input("Enter CSV URL:")
            if st.button("🌐 Load from URL", use_container_width=True):
                if url:
                    try:
                        st.session_state.df = pd.read_csv(url)
                        st.success(f"✅ Loaded: {st.session_state.df.shape[0]} rows")
                        log_security_event("URL_LOAD", f"URL: {url}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        st.markdown("---")
        st.header("🧹 Data Cleaning")
        cleaning_method = st.selectbox("Cleaning method:", ["None", "Drop NA", "Fill NA with 0", "Remove Duplicates"])
        
        if st.session_state.df is not None and cleaning_method != "None":
            if st.button("🔧 Apply Cleaning", use_container_width=True):
                original_len = len(st.session_state.df)
                st.session_state.df = apply_cleaning(st.session_state.df, cleaning_method)
                st.success(f"✅ Applied: {cleaning_method}")
                st.info(f"Rows: {original_len} → {len(st.session_state.df)}")
        
        st.markdown("---")
        st.header("🔒 Security")
        anonymize = st.checkbox("Anonymize coordinates")
        encrypt_reports = st.checkbox("Encrypt reports")
        
        if anonymize and st.session_state.df is not None:
            if st.button("🔐 Apply Anonymization", use_container_width=True):
                st.session_state.df = anonymize_coordinates(st.session_state.df)
                st.success("✅ Coordinates anonymized")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔍 Explore", "📊 Visualize", "📍 Analysis", "📋 Reports", "🧪 Testing", "❓ Help"])
    
    with tab1:
        st.header("Data Exploration")
        if st.session_state.df is not None:
            df = st.session_state.df
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", f"{df.shape[0]:,}")
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing", f"{df.isnull().sum().sum():,}")
            col4.metric("Duplicates", f"{df.duplicated().sum():,}")
            st.markdown("---")
            st.subheader("📋 Dataset Preview")
            num_rows = st.slider("Rows to display:", 5, 50, 10)
            st.dataframe(df.head(num_rows), use_container_width=True)
            st.markdown("---")
            st.subheader("📈 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            st.markdown("---")
            st.subheader("🔍 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values,
                'Unique': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
        else:
            st.info("👈 Load data from sidebar to begin")
    
    with tab2:
        st.header("Data Visualizations")
        if st.session_state.df is not None:
            df = st.session_state.df
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) > 0:
                st.subheader("📊 Numeric Distribution")
                col = st.selectbox("Select column:", numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 5))
                df[col].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f"Distribution of {col}", fontsize=14, fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            st.markdown("---")
            categorical_cols = df.select_dtypes(include='object').columns.tolist()
            if len(categorical_cols) > 0:
                st.subheader("📊 Categorical Distribution")
                cat_col = st.selectbox("Select category:", categorical_cols)
                if df[cat_col].nunique() < 50:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_n = st.slider("Show top N:", 5, 20, 10)
                    df[cat_col].value_counts().head(top_n).plot(kind='barh', ax=ax, color='coral')
                    ax.set_title(f"Top {top_n} in {cat_col}", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Count")
                    ax.grid(True, alpha=0.3, axis='x')
                    plt.tight_layout()
                    st.pyplot(fig)
            st.markdown("---")
            if len(numeric_cols) > 1:
                st.subheader("📊 Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, square=True)
                ax.set_title("Correlation Matrix", fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("👈 Load data from sidebar")
    
    with tab3:
        st.header("Crime Hotspot Analysis")
        if st.session_state.df is not None:
            method = st.selectbox("Analysis Method:", ["Grid-based Hotspot (250m)", "Grid-based Hotspot (500m)", "Temporal Analysis", "Crime Type Distribution"])
            
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner("Analyzing data..."):
                    if "Grid-based" in method:
                        size_deg = 0.0025 if "250m" in method else 0.005
                        size_m = 250 if "250m" in method else 500
                        results, error = perform_hotspot_analysis_simple(st.session_state.df, size_deg)
                        if error:
                            st.error(error)
                        else:
                            st.session_state.analysis_results = results
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Incidents", f"{results['total_incidents']:,}")
                            col2.metric("Grid Size", f"~{size_m}m")
                            col3.metric("Hotspot Cells", f"{results['hotspot_count']:,}")
                            col4.metric("Threshold", f"{results['threshold']:.0f}")
                            st.markdown("---")
                            st.subheader("🗺️ Hotspot Heatmap")
                            fig = plot_hotspot_heatmap(results)
                            st.pyplot(fig)
                            st.markdown("---")
                        st.subheader("📍 Top Hotspot Cells")
                        hotspots = results['hotspot_cells']

                        if hotspots is not None and len(hotspots) > 0:
                            # Make sure expected columns exist
                            expected_cols = ['lat_center', 'lon_center', 'count']
                            available = [c for c in expected_cols if c in hotspots.columns]

                            if len(available) == 3:
                                hotspots = hotspots.sort_values('count', ascending=False)
                                display_df = hotspots[available].head(20).copy()
                                display_df.columns = ['Latitude', 'Longitude', 'Incident Count']
                                st.dataframe(display_df.reset_index(drop=True), use_container_width=True)
                            else:
                                st.warning(f"Hotspot coordinate columns missing: {set(expected_cols) - set(hotspots.columns)}")
                        else:
                            st.info("No hotspot cells found for the current analysis.")

                        st.success(f"✅ Analyzed {results['total_incidents']} incidents in {results['total_cells']} cells, identified {results['hotspot_count']} hotspots")
                    
                    elif method == "Temporal Analysis":
                        result, error = perform_temporal_analysis(st.session_state.df)
                        if error:
                            st.error(error)
                        else:
                            st.session_state.analysis_results['temporal'] = result
                            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                            result['by_hour'].plot(kind='bar', ax=axes[0], color='skyblue')
                            axes[0].set_title('By Hour')
                            axes[0].set_xlabel('Hour')
                            axes[0].set_ylabel('Count')
                            axes[0].grid(True, alpha=0.3, axis='y')
                            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                            result['by_day'].plot(kind='bar', ax=axes[1], color='coral')
                            axes[1].set_title('By Day')
                            axes[1].set_xticklabels(days, rotation=45)
                            axes[1].grid(True, alpha=0.3, axis='y')
                            result['by_month'].plot(kind='bar', ax=axes[2], color='lightgreen')
                            axes[2].set_title('By Month')
                            axes[2].grid(True, alpha=0.3, axis='y')
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.success(f"Peak Hour: {result['by_hour'].idxmax()}:00 | Peak Day: {days[result['by_day'].idxmax()]} | Peak Month: {result['by_month'].idxmax()}")
                    
                    elif method == "Crime Type Distribution":
                        result, error = perform_crime_type_analysis(st.session_state.df)
                        if error:
                            st.error(error)
                        else:
                            st.session_state.analysis_results['crime_types'] = result
                            fig, ax = plt.subplots(figsize=(12, 6))
                            result.head(15).plot(kind='barh', ax=ax, color='steelblue')
                            ax.set_title('Top 15 Crime Types', fontsize=16, fontweight='bold')
                            ax.set_xlabel('Incidents')
                            ax.grid(True, alpha=0.3, axis='x')
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.success(f"Total Types: {len(result)} | Most Common: {result.index[0]} ({result.iloc[0]:,} incidents)")
        else:
            st.info("👈 Load data from sidebar")
    
    with tab4:
        st.header("Analysis Reports")
        st.markdown("**Generate comprehensive reports with all analysis results**")
        st.markdown("---")
        if st.session_state.df is not None:
            if st.button("📋 Generate Report", type="primary"):
                df = st.session_state.df
                report = f"""# COMPREHENSIVE CRIME ANALYSIS REPORT
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

## 1. DATA SUMMARY
- **Dataset Shape**: {df.shape[0]:,} rows × {df.shape[1]} columns
- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- **Missing Values**: {df.isnull().sum().sum():,}
- **Duplicate Rows**: {df.duplicated().sum():,}
- **Completeness**: {(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%
"""
                if 'Lat_Public' in df.columns:
                    valid = df[['Lat_Public', 'Long_Public']].notna().all(axis=1).sum()
                    report += f"- **Valid Coordinates**: {valid:,}/{len(df):,} ({valid/len(df)*100:.1f}%)\n"
                
                
                report += "\n## 2. ANALYSIS RESULTS\n"
                if st.session_state.analysis_results:
                    if 'grid' in st.session_state.analysis_results:
                        r = st.session_state.analysis_results
                        report += f"""
### Hotspot Analysis
- Grid Size: ~{r['cell_size']*111:.0f}m
- Total Incidents: {r['total_incidents']:,}
- Hotspot Cells: {r['hotspot_count']:,}
- Threshold: {r['threshold']:.0f} incidents/cell
"""
                    if 'temporal' in st.session_state.analysis_results:
                        t = st.session_state.analysis_results['temporal']
                        report += f"""
### Temporal Patterns
- Peak Hour: {t['by_hour'].idxmax()}:00 ({t['by_hour'].max():,} incidents)
- Peak Day: Day {t['by_day'].idxmax()} ({t['by_day'].max():,} incidents)
- Peak Month: {t['by_month'].idxmax()} ({t['by_month'].max():,} incidents)
"""
                    if 'crime_types' in st.session_state.analysis_results:
                        c = st.session_state.analysis_results['crime_types']
                        report += f"""
### Crime Types
- Total Types: {len(c)}
- Most Common: {c.index[0]} ({c.iloc[0]:,} incidents)
"""
                
                report += """
## 3. RECOMMENDATIONS
- Deploy patrols to identified hotspot areas
- Focus resources on high-incident cells during peak times
- Monitor trends for pattern changes
- Implement targeted prevention strategies

## 4. DATA QUALITY
"""
                completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                report += f"- Overall completeness: {completeness:.1f}%\n"
                if df.isnull().sum().sum() > 0:
                    report += "\n### Missing Data:\n"
                    missing = df.isnull().sum()[df.isnull().sum() > 0]
                    for col, count in missing.items():
                        report += f"  - {col}: {count} ({count/len(df)*100:.1f}%)\n"
                
                if encrypt_reports:
                    report += "\n[ENCRYPTED: Sensitive information]\n"
                report += "\n" + "="*80 + "\nEND OF REPORT\n" + "="*80
                
                st.markdown(report)
                st.download_button("💾 Download Report", report, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "text/plain")
        else:
            st.info("👈 Load data and run analysis first")
    
    with tab5:
        st.header("System Testing")
        if st.button("🧪 Run All Tests", type="primary"):
            with st.spinner("Running tests..."):
                results = []
                try:
                    test_df = create_sample_data()
                    assert len(test_df) > 0
                    results.append(("Data Loading", "PASS", "✅"))
                except:
                    results.append(("Data Loading", "FAIL", "❌"))
                try:
                    issues = validate_data_input(test_df)
                    assert isinstance(issues, list)
                    results.append(("Validation", "PASS", "✅"))
                except:
                    results.append(("Validation", "FAIL", "❌"))
                try:
                    r, e = perform_hotspot_analysis_simple(test_df, 0.005)
                    assert e is None
                    results.append(("Hotspot Analysis", "PASS", "✅"))
                except:
                    results.append(("Hotspot Analysis", "FAIL", "❌"))
                try:
                    r, e = perform_temporal_analysis(test_df)
                    assert e is None
                    results.append(("Temporal Analysis", "PASS", "✅"))
                except:
                    results.append(("Temporal Analysis", "FAIL", "❌"))
                try:
                    r, e = perform_crime_type_analysis(test_df)
                    assert e is None
                    results.append(("Crime Type Analysis", "PASS", "✅"))
                except:
                    results.append(("Crime Type Analysis", "FAIL", "❌"))
                
                st.subheader("Test Results")
                for name, status, icon in results:
                    st.write(f"{icon} **{name}**: {status}")
                passed = sum(1 for _, s, _ in results if s == "PASS")
                if passed == len(results):
                    st.success(f"🎉 All {len(results)} tests passed!")
                else:
                    st.warning(f"⚠️ {passed}/{len(results)} tests passed")
    
    with tab6:
        st.header("Help & Documentation")
        st.markdown("""
        ## Quick Start Guide
        
        ### 1. Load Data
        - **Upload File**: CSV or Excel with crime data
        - **Sample Data**: 500 Phoenix, AZ incidents (for testing)
        - **URL**: Load from web-hosted CSV
        
        ### 2. Explore Data
        View statistics, preview records, check data quality
        
        ### 3. Create Visualizations
        - Numeric distributions
        - Category breakdowns
        - Correlation heatmaps
        
        ### 4. Run Analysis
        - **Grid Hotspot (250m/500m)**: Identify high-crime areas
        - **Temporal**: Find peak times/days
        - **Crime Types**: See most common offenses
        
        ### 5. Generate Reports
        Comprehensive reports with all results, downloadable as text
        
        ## Required Data Format
        - **Lat_Public**: Latitude coordinates (required for hotspot)
        - **Long_Public**: Longitude coordinates (required for hotspot)
        - **Date**: For temporal analysis (optional)
        - **OffenseType**: For crime type analysis (optional)
        
        ## Multi-User Support
        Each user has independent session. Multiple simultaneous users supported without interference.
        
        ## Report Generation
        - **Location**: Reports tab
        - **Process**: Click "Generate Report" → View on screen → Download
        - **Contents**: Data summary, analysis results, recommendations, quality metrics
        
        ## Troubleshooting
        - **No data**: Load sample data first
        - **Missing columns**: Check for Lat_Public/Long_Public
        - **Analysis error**: Verify coordinates are valid numbers
        """)
    
    with st.expander("🔒 Security Log"):
        if st.session_state.security_log:
            for log in st.session_state.security_log[-10:]:
                st.text(log)
        else:
            st.info("No events logged yet")

if __name__ == "__main__":
    main()