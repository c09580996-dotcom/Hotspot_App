"""
Crime Hotspot Analysis Tool - Streamlit Web Application
Author: Chynara Gambarova
CST-580 | December 2025

COMPLETE WORKING VERSION - Ready for deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point, box
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crime Hotspot Analysis",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'security_log' not in st.session_state:
    st.session_state.security_log = []

# ============================================================================
# SECURITY FUNCTIONS
# ============================================================================

def log_security_event(event_type, details):
    """Log security-related events"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.security_log.append(f"[{timestamp}] {event_type}: {details}")

def validate_data_input(data):
    """Validate uploaded data for security concerns"""
    log_security_event("VALIDATION", "Checking uploaded data")
    issues = []
    
    if data is not None:
        suspicious_cols = [col for col in data.columns if
                          any(x in str(col).lower() for x in ['password', 'ssn', 'credit'])]
        if suspicious_cols:
            issues.append(f"Warning: Sensitive column names detected: {suspicious_cols}")
        
        if len(data) > 1000000:
            issues.append("Warning: Large dataset (>1M rows) may cause performance issues")
    
    return issues

def anonymize_coordinates(df):
    """Add random noise to coordinates for privacy"""
    df_copy = df.copy()
    if 'Lat_Public' in df_copy.columns and 'Long_Public' in df_copy.columns:
        df_copy['Lat_Public'] = df_copy['Lat_Public'] + np.random.normal(0, 0.001, len(df_copy))
        df_copy['Long_Public'] = df_copy['Long_Public'] + np.random.normal(0, 0.001, len(df_copy))
        log_security_event("ANONYMIZATION", "Coordinates anonymized with noise")
    return df_copy

# ============================================================================
# DATA FUNCTIONS
# ============================================================================

def create_sample_data():
    """Create sample crime incident data for testing"""
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
    """Apply data cleaning operations"""
    if df is None or method == 'None':
        return df
    
    df_copy = df.copy()
    original_shape = df_copy.shape
    
    if method == 'Drop NA':
        df_copy = df_copy.dropna()
        log_security_event("CLEANING", f"Dropped NA values: {original_shape[0] - df_copy.shape[0]} rows removed")
    elif method == 'Fill NA with 0':
        df_copy = df_copy.fillna(0)
        log_security_event("CLEANING", "Filled NA values with 0")
    elif method == 'Remove Duplicates':
        df_copy = df_copy.drop_duplicates()
        log_security_event("CLEANING", f"Removed duplicates: {original_shape[0] - df_copy.shape[0]} rows removed")
    
    return df_copy

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def perform_hotspot_analysis(df, cell_size=250):
    """Perform grid-based hotspot analysis"""
    lat_col = "Lat_Public"
    lon_col = "Long_Public"
    
    if lat_col not in df.columns or lon_col not in df.columns:
        return None, "❌ Missing Lat_Public or Long_Public columns."
    
    df_clean = df.dropna(subset=[lat_col, lon_col])
    if len(df_clean) == 0:
        return None, "❌ No valid coordinates found."
    
    gdf = gpd.GeoDataFrame(
        df_clean,
        geometry=gpd.points_from_xy(df_clean[lon_col], df_clean[lat_col]),
        crs="EPSG:4326"
    ).to_crs(3857)
    
    minx, miny, maxx, maxy = gdf.total_bounds
    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)
    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]
    grid = gpd.GeoDataFrame(geometry=cells, crs=gdf.crs)
    
    joined = gpd.sjoin(gdf, grid, how="left", predicate="intersects")
    counts = joined.groupby("index_right").size()
    
    grid["count"] = counts
    grid["count"] = grid["count"].fillna(0).astype(int)
    
    incident_cells = grid[grid["count"] > 0].copy()
    
    if len(incident_cells) > 0:
        incident_cells = incident_cells.sort_values("count", ascending=False)
        k = max(1, int(len(incident_cells) * 0.20))
        hotspot_indices = incident_cells.index[:k]
        grid["hotspot"] = grid.index.isin(hotspot_indices)
        threshold = incident_cells["count"].iloc[k - 1]
    else:
        threshold = 0
        grid["hotspot"] = False
    
    results = {
        'grid': grid,
        'threshold': threshold,
        'total_incidents': len(gdf),
        'total_cells': len(grid),
        'incident_cells': len(incident_cells),
        'hotspot_cells': grid["hotspot"].sum(),
        'cell_size': cell_size
    }
    
    return results, None

def perform_temporal_analysis(df):
    """Analyze incidents by time patterns"""
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
        return None, "❌ No date/time column found."
    
    df['hour'] = df[date_col].dt.hour
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    
    return {
        'by_hour': df.groupby('hour').size(),
        'by_day': df.groupby('day_of_week').size(),
        'by_month': df.groupby('month').size()
    }, None

def perform_crime_type_analysis(df):
    """Analyze distribution of crime types"""
    crime_col = None
    for col in df.columns:
        if any(x in col.lower() for x in ['offense', 'crime', 'type', 'category']):
            crime_col = col
            break
    
    if crime_col is None:
        return None, "❌ No crime type column found."
    
    return df[crime_col].value_counts(), None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🚔 Crime Hotspot Analysis Tool")
    st.markdown("**Comprehensive Data Product for Law Enforcement Analysis**")
    st.markdown("*Author: Chynara Gambarova | CST-580 | December 2025*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Management")
        
        load_option = st.radio(
            "Select Data Source:",
            ["Upload File", "Load Sample Data", "Load from URL"]
        )
        
        if load_option == "Upload File":
            uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        st.session_state.df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        st.session_state.df = pd.read_excel(uploaded_file)
                    
                    issues = validate_data_input(st.session_state.df)
                    st.success(f"✅ Loaded: {st.session_state.df.shape[0]} rows")
                    for issue in issues:
                        st.warning(issue)
                    log_security_event("UPLOAD", f"File: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        elif load_option == "Load Sample Data":
            if st.button("📊 Load Sample Dataset"):
                st.session_state.df = create_sample_data()
                st.success("✅ Sample data loaded (500 records)")
                log_security_event("DATA_LOAD", "Sample dataset loaded")
        
        elif load_option == "Load from URL":
            url = st.text_input("Enter CSV URL:")
            if st.button("🌐 Load from URL"):
                if url:
                    try:
                        st.session_state.df = pd.read_csv(url)
                        st.success(f"✅ Loaded: {st.session_state.df.shape[0]} rows")
                        log_security_event("URL_LOAD", f"URL: {url}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        st.markdown("---")
        st.header("🧹 Data Cleaning")
        cleaning_method = st.selectbox(
            "Cleaning method:",
            ["None", "Drop NA", "Fill NA with 0", "Remove Duplicates"]
        )
        
        if st.session_state.df is not None and cleaning_method != "None":
            if st.button("🔧 Apply Cleaning"):
                st.session_state.df = apply_cleaning(st.session_state.df, cleaning_method)
                st.success(f"✅ Applied: {cleaning_method}")
        
        st.markdown("---")
        st.header("🔒 Security")
        anonymize = st.checkbox("Anonymize coordinates")
        encrypt_reports = st.checkbox("Encrypt reports")
        
        if anonymize and st.session_state.df is not None:
            if st.button("🔐 Anonymize"):
                st.session_state.df = anonymize_coordinates(st.session_state.df)
                st.success("✅ Anonymized")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Explore", "📊 Visualize", "📍 Analysis", "📋 Reports", "🧪 Testing", "❓ Help"
    ])
    
    # Tab 1: Explore
    with tab1:
        st.header("Data Exploration")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", f"{df.shape[0]:,}")
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing", f"{df.isnull().sum().sum():,}")
            col4.metric("Duplicates", f"{df.duplicated().sum():,}")
            
            st.subheader("📋 Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("📈 Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        else:
            st.info("👈 Load data from sidebar")
    
    # Tab 2: Visualize
    with tab2:
        st.header("Visualizations")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                st.subheader("📊 Distribution")
                col = st.selectbox("Select column:", numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 5))
                df[col].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f"Distribution of {col}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            if len(numeric_cols) > 1:
                st.subheader("📊 Correlation")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
        else:
            st.info("👈 Load data from sidebar")
    
    # Tab 3: Analysis
    with tab3:
        st.header("Hotspot Analysis")
        
        if st.session_state.df is not None:
            method = st.selectbox(
                "Analysis Method:",
                ["Grid-based Hotspot (250m)", "Grid-based Hotspot (500m)", 
                 "Temporal Analysis", "Crime Type Distribution"]
            )
            
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner("Analyzing..."):
                    
                    if "Grid-based" in method:
                        size = 250 if "250m" in method else 500
                        results, error = perform_hotspot_analysis(st.session_state.df, size)
                        
                        if error:
                            st.error(error)
                        else:
                            st.session_state.analysis_results = results
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Incidents", results['total_incidents'])
                            col2.metric("Grid Size", f"{size}m")
                            col3.metric("Hotspot Cells", results['hotspot_cells'])
                            col4.metric("Threshold", f"{results['threshold']:.0f}")
                            
                            st.subheader("🗺️ Hotspot Map")
                            fig, ax = plt.subplots(figsize=(12, 10))
                            results['grid'].plot(
                                column="count", cmap="YlOrRd", linewidth=0.1,
                                ax=ax, edgecolor="gray", legend=True
                            )
                            results['grid'][results['grid']["hotspot"]].boundary.plot(
                                ax=ax, color="black", linewidth=0.8
                            )
                            ax.set_title(f"Hotspot Analysis - {size}m Grid", fontsize=16, fontweight="bold")
                            st.pyplot(fig)
                            
                            st.subheader("📍 Top Hotspots")
                            hotspots = results['grid'][results['grid']["hotspot"]].sort_values("count", ascending=False)
                            st.dataframe(hotspots.head(20)[["count"]].reset_index(drop=True))
                    
                    elif method == "Temporal Analysis":
                        result, error = perform_temporal_analysis(st.session_state.df)
                        if error:
                            st.error(error)
                        else:
                            st.session_state.analysis_results['temporal'] = result
                            
                            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                            
                            result['by_hour'].plot(kind='bar', ax=axes[0], color='skyblue')
                            axes[0].set_title('By Hour')
                            axes[0].grid(True, alpha=0.3, axis='y')
                            
                            result['by_day'].plot(kind='bar', ax=axes[1], color='coral')
                            axes[1].set_title('By Day')
                            axes[1].grid(True, alpha=0.3, axis='y')
                            
                            result['by_month'].plot(kind='bar', ax=axes[2], color='lightgreen')
                            axes[2].set_title('By Month')
                            axes[2].grid(True, alpha=0.3, axis='y')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.success(f"""
                            📊 Peak Hour: {result['by_hour'].idxmax()}:00 ({result['by_hour'].max()} incidents)
                            📊 Peak Day: {result['by_day'].idxmax()} ({result['by_day'].max()} incidents)
                            📊 Peak Month: {result['by_month'].idxmax()} ({result['by_month'].max()} incidents)
                            """)
                    
                    elif method == "Crime Type Distribution":
                        result, error = perform_crime_type_analysis(st.session_state.df)
                        if error:
                            st.error(error)
                        else:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            result.head(15).plot(kind='barh', ax=ax, color='steelblue')
                            ax.set_title('Top 15 Crime Types', fontsize=16)
                            ax.grid(True, alpha=0.3, axis='x')
                            plt.tight_layout()
                            st.pyplot(fig)
        else:
            st.info("👈 Load data from sidebar")
    
    # Tab 4: Reports
    with tab4:
        st.header("Analysis Reports")
        
        if st.session_state.df is not None:
            if st.button("📋 Generate Report", type="primary"):
                report = f"""# CRIME ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## DATA SUMMARY
- Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns
- Missing: {st.session_state.df.isnull().sum().sum()}
- Duplicates: {st.session_state.df.duplicated().sum()}

## ANALYSIS RESULTS
"""
                if st.session_state.analysis_results:
                    if 'grid' in st.session_state.analysis_results:
                        r = st.session_state.analysis_results
                        report += f"""
### Hotspot Analysis
- Cell Size: {r['cell_size']}m
- Total Incidents: {r['total_incidents']}
- Hotspot Cells: {r['hotspot_cells']}
- Threshold: {r['threshold']:.0f}
"""
                    if 'temporal' in st.session_state.analysis_results:
                        t = st.session_state.analysis_results['temporal']
                        report += f"""
### Temporal Patterns
- Peak Hour: {t['by_hour'].idxmax()}:00
- Peak Day: {t['by_day'].idxmax()}
"""
                
                report += """
## RECOMMENDATIONS
- Deploy patrols to hotspot areas
- Focus on high-incident cells
- Monitor trends over time

---
END OF REPORT
"""
                st.markdown(report)
                
                st.download_button(
                    "💾 Download Report",
                    report,
                    f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain"
                )
        else:
            st.info("👈 Load data and run analysis first")
    
    # Tab 5: Testing
    with tab5:
        st.header("System Testing")
        
        if st.button("🧪 Run All Tests", type="primary"):
            with st.spinner("Running tests..."):
                results = []
                
                # Test 1
                try:
                    test_df = create_sample_data()
                    assert len(test_df) > 0
                    results.append(("Data Loading", "PASS", "✅"))
                except:
                    results.append(("Data Loading", "FAIL", "❌"))
                
                # Test 2
                try:
                    issues = validate_data_input(test_df)
                    assert isinstance(issues, list)
                    results.append(("Validation", "PASS", "✅"))
                except:
                    results.append(("Validation", "FAIL", "❌"))
                
                # Test 3
                try:
                    r, e = perform_hotspot_analysis(test_df, 250)
                    assert e is None
                    results.append(("Hotspot Analysis", "PASS", "✅"))
                except:
                    results.append(("Hotspot Analysis", "FAIL", "❌"))
                
                # Test 4
                try:
                    r, e = perform_temporal_analysis(test_df)
                    assert e is None
                    results.append(("Temporal Analysis", "PASS", "✅"))
                except:
                    results.append(("Temporal Analysis", "FAIL", "❌"))
                
                st.subheader("Test Results")
                for name, status, icon in results:
                    st.write(f"{icon} **{name}**: {status}")
                
                passed = sum(1 for _, s, _ in results if s == "PASS")
                if passed == len(results):
                    st.success(f"🎉 All {len(results)} tests passed!")
                else:
                    st.warning(f"⚠️ {passed}/{len(results)} tests passed")
    
    # Tab 6: Help
    with tab6:
        st.header("Help & Documentation")
        st.markdown("""
        ## Quick Start Guide
        
        ### 1. Load Data
        Use the sidebar to:
        - Upload your CSV/Excel file
        - Load sample data (500 records)
        - Load from URL
        
        ### 2. Explore
        View dataset statistics, preview, and quality metrics
        
        ### 3. Visualize
        Create charts and correlation heatmaps
        
        ### 4. Analyze
        Run hotspot analysis:
        - **Grid-based (250m/500m)**: Identify high-crime cells
        - **Temporal**: Patterns by time
        - **Crime Type**: Distribution analysis
        
        ### 5. Generate Reports
        Create comprehensive reports with download option
        
        ### Required Data Format
        - **Columns**: Lat_Public, Long_Public (required)
        - **Optional**: Date, OffenseType, District
        - **Formats**: CSV, Excel
        
        ### Multi-User Support
        Each user has independent session. Multiple users can access
        simultaneously without data interference.
        
        ### Report Generation
        - Located in **📋 Reports** tab
        - Click "Generate Report" button
        - View in-page or download as text file
        - Includes all analysis results
        
        ### Troubleshooting
        - **No data loaded**: Use sidebar to load data first
        - **Missing columns**: Check for Lat_Public/Long_Public
        - **Analysis error**: Ensure valid coordinates exist
        """)
    
    # Footer
    with st.expander("🔒 Security Log"):
        if st.session_state.security_log:
            for log in st.session_state.security_log[-10:]:
                st.text(log)
        else:
            st.info("No events logged yet")

if __name__ == "__main__":
    main()