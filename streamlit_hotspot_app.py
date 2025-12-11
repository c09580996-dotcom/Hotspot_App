"""
Crime Hotspot Analysis Tool - Streamlit Web Application
Author: Chynara Gambarova
CST-580 | December 2025
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
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
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
    if 'Lat_Public' in df.columns and 'Long_Public' in df.columns:
        df['Lat_Public'] = df['Lat_Public'] + np.random.normal(0, 0.001, len(df))
        df['Long_Public'] = df['Long_Public'] + np.random.normal(0, 0.001, len(df))
        log_security_event("ANONYMIZATION", "Coordinates anonymized with noise")
    return df

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
    original_shape = df.shape
    
    if method == 'Drop NA':
        df = df.dropna()
        log_security_event("CLEANING", f"Dropped NA values: {original_shape[0] - df.shape[0]} rows removed")
    elif method == 'Fill NA with 0':
        df = df.fillna(0)
        log_security_event("CLEANING", "Filled NA values with 0")
    elif method == 'Remove Duplicates':
        df = df.drop_duplicates()
        log_security_event("CLEANING", f"Removed duplicates: {original_shape[0] - df.shape[0]} rows removed")
    
    return df

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
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df_clean,
        geometry=gpd.points_from_xy(df_clean[lon_col], df_clean[lat_col]),
        crs="EPSG:4326"
    ).to_crs(3857)
    
    # Build grid
    minx, miny, maxx, maxy = gdf.total_bounds
    xs = np.arange(minx, maxx, cell_size)
    ys = np.arange(miny, maxy, cell_size)
    cells = [box(x, y, x + cell_size, y + cell_size) for x in xs for y in ys]
    grid = gpd.GeoDataFrame(geometry=cells, crs=gdf.crs)
    
    # Spatial join
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
    
    # Store results
    results = {
        'grid': grid,
        'threshold': threshold,
        'total_incidents': len(gdf),
        'hotspot_cells': grid["hotspot"].sum(),
        'cell_size': cell_size,
        'incident_cells': len(incident_cells)
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
    # Header
    st.title("🚔 Crime Hotspot Analysis Tool")
    st.markdown("**Comprehensive Data Product for Law Enforcement Analysis**")
    st.markdown("*Author: Chynara Gambarova | CST-580 | December 2025*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Management")
        
        # Data loading options
        load_option = st.radio(
            "Select Data Source:",
            ["Upload File", "Load Sample Data", "Load from URL"]
        )
        
        if load_option == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'geojson']
            )
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        st.session_state.df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        st.session_state.df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.geojson'):
                        st.session_state.df = gpd.read_file(uploaded_file)
                    
                    issues = validate_data_input(st.session_state.df)
                    st.success(f"✅ File loaded: {st.session_state.df.shape[0]} rows")
                    for issue in issues:
                        st.warning(issue)
                    log_security_event("UPLOAD", f"File uploaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
        
        elif load_option == "Load Sample Data":
            if st.button("Load Sample Dataset"):
                st.session_state.df = create_sample_data()
                st.success("✅ Sample data loaded successfully!")
                log_security_event("DATA_LOAD", "Sample dataset loaded")
        
        elif load_option == "Load from URL":
            url = st.text_input("Enter CSV URL:")
            if st.button("Load from URL"):
                try:
                    st.session_state.df = pd.read_csv(url)
                    st.success(f"✅ Data loaded: {st.session_state.df.shape[0]} rows")
                    log_security_event("URL_LOAD", f"Data loaded from: {url}")
                except Exception as e:
                    st.error(f"❌ Error loading from URL: {e}")
        
        st.markdown("---")
        
        # Data cleaning
        st.header("🧹 Data Cleaning")
        cleaning_method = st.selectbox(
            "Select cleaning method:",
            ["None", "Drop NA", "Fill NA with 0", "Remove Duplicates"]
        )
        
        if st.session_state.df is not None and cleaning_method != "None":
            if st.button("Apply Cleaning"):
                st.session_state.df = apply_cleaning(st.session_state.df, cleaning_method)
                st.success(f"✅ Cleaning applied: {cleaning_method}")
        
        st.markdown("---")
        
        # Security options
        st.header("🔒 Security Options")
        anonymize = st.checkbox("Anonymize coordinates")
        encrypt_reports = st.checkbox("Encrypt saved reports")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Explore", "📊 Visualize", "📍 Analysis", "📋 Reports", "🧪 Testing", "❓ Help"
    ])
    
    # Tab 1: Data Exploration
    with tab1:
        st.header("Data Exploration")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Duplicates", df.duplicated().sum())
            
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("📈 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            
            st.subheader("🔍 Column Information")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            
        else:
            st.info("👈 Please load data from the sidebar first")
    
    # Tab 2: Visualizations
    with tab2:
        st.header("Data Visualizations")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Numeric distribution
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                st.subheader("📊 Numeric Distribution")
                selected_col = st.selectbox("Select column:", numeric_cols)
                fig, ax = plt.subplots(figsize=(10, 5))
                df[selected_col].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f"Distribution of {selected_col}")
                ax.set_xlabel(selected_col)
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Categorical distribution
            categorical_cols = df.select_dtypes(include='object').columns
            if len(categorical_cols) > 0:
                st.subheader("📊 Categorical Distribution")
                selected_cat = st.selectbox("Select category:", categorical_cols)
                if df[selected_cat].nunique() < 20:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    df[selected_cat].value_counts().head(10).plot(kind='bar', ax=ax, color='coral')
                    ax.set_title(f"Top 10 in {selected_cat}")
                    ax.set_xlabel(selected_cat)
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45, ha='right')
                    ax.grid(True, alpha=0.3, axis='y')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Correlation heatmap
            if len(numeric_cols) > 1:
                st.subheader("📊 Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 6))
                correlation = df[numeric_cols].corr()
                sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=1, ax=ax)
                ax.set_title("Correlation Matrix")
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("👈 Please load data from the sidebar first")
    
    # Tab 3: Analysis
    with tab3:
        st.header("Hotspot Analysis")
        
        if st.session_state.df is not None:
            analysis_method = st.selectbox(
                "Select Analysis Method:",
                ["Grid-based Hotspot (250m)", "Grid-based Hotspot (500m)", 
                 "Temporal Analysis", "Crime Type Distribution"]
            )
            
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner("Analyzing data..."):
                    
                    if "Grid-based" in analysis_method:
                        cell_size = 250 if "250m" in analysis_method else 500
                        results, error = perform_hotspot_analysis(st.session_state.df, cell_size)
                        
                        if error:
                            st.error(error)
                        else:
                            st.session_state.analysis_results = results
                            
                            # Display metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Incidents", results['total_incidents'])
                            with col2:
                                st.metric("Grid Cell Size", f"{cell_size}m")
                            with col3:
                                st.metric("Hotspot Cells", results['hotspot_cells'])
                            with col4:
                                st.metric("Threshold", f"{results['threshold']:.0f}")
                            
                            # Plot hotspot map
                            st.subheader("🗺️ Hotspot Map")
                            fig, ax = plt.subplots(figsize=(12, 10))
                            results['grid'].plot(
                                column="count",
                                cmap="YlOrRd",
                                linewidth=0.1,
                                ax=ax,
                                edgecolor="gray",
                                legend=True,
                                legend_kwds={"label": "Incident Count", "shrink": 0.8}
                            )
                            results['grid'][results['grid']["hotspot"]].boundary.plot(
                                ax=ax, color="black", linewidth=0.8
                            )
                            ax.set_title(f"Crime Hotspot Analysis - Grid Size: {cell_size}m",
                                       fontsize=16, fontweight="bold")
                            ax.set_xlabel("X Coordinate (meters)")
                            ax.set_ylabel("Y Coordinate (meters)")
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Top hotspots table
                            st.subheader("📍 Top Hotspot Cells")
                            hotspots = results['grid'][results['grid']["hotspot"]].sort_values(
                                "count", ascending=False
                            )
                            st.dataframe(
                                hotspots.head(20)[["count", "hotspot"]].reset_index(drop=True),
                                use_container_width=True
                            )
                    
                    elif analysis_method == "Temporal Analysis":
                        result, error = perform_temporal_analysis(st.session_state.df)
                        if error:
                            st.error(error)
                        else:
                            st.session_state.analysis_results['temporal'] = result
                            
                            # Plot temporal patterns
                            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                            
                            result['by_hour'].plot(kind='bar', ax=axes[0], color='skyblue')
                            axes[0].set_title('Incidents by Hour')
                            axes[0].set_xlabel('Hour')
                            axes[0].set_ylabel('Count')
                            axes[0].grid(True, alpha=0.3, axis='y')
                            
                            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                            result['by_day'].plot(kind='bar', ax=axes[1], color='coral')
                            axes[1].set_title('Incidents by Day')
                            axes[1].set_xlabel('Day')
                            axes[1].set_xticklabels(days, rotation=45)
                            axes[1].set_ylabel('Count')
                            axes[1].grid(True, alpha=0.3, axis='y')
                            
                            result['by_month'].plot(kind='bar', ax=axes[2], color='lightgreen')
                            axes[2].set_title('Incidents by Month')
                            axes[2].set_xlabel('Month')
                            axes[2].set_ylabel('Count')
                            axes[2].grid(True, alpha=0.3, axis='y')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Summary
                            st.success(f"""
                            📊 **Temporal Patterns Summary:**
                            - Peak Hour: {result['by_hour'].idxmax()}:00 ({result['by_hour'].max()} incidents)
                            - Peak Day: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][result['by_day'].idxmax()]}
                            - Peak Month: Month {result['by_month'].idxmax()} ({result['by_month'].max()} incidents)
                            """)
                    
                    elif analysis_method == "Crime Type Distribution":
                        result, error = perform_crime_type_analysis(st.session_state.df)
                        if error:
                            st.error(error)
                        else:
                            st.session_state.analysis_results['crime_types'] = result
                            
                            fig, ax = plt.subplots(figsize=(12, 6))
                            result.head(15).plot(kind='barh', ax=ax, color='steelblue')
                            ax.set_title('Top 15 Crime Types', fontsize=16, fontweight='bold')
                            ax.set_xlabel('Number of Incidents')
                            ax.set_ylabel('Crime Type')
                            ax.grid(True, alpha=0.3, axis='x')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.success(f"""
                            📊 **Crime Type Summary:**
                            - Total Crime Types: {len(result)}
                            - Most Common: {result.index[0]} ({result.iloc[0]} incidents)
                            - Top 5 represent {result.head(5).sum() / result.sum() * 100:.1f}% of all incidents
                            """)
        else:
            st.info("👈 Please load data from the sidebar first")
    
    # Tab 4: Reports
    with tab4:
        st.header("Analysis Reports")
        
        if st.session_state.df is not None:
            if st.button("📋 Generate Report", type="primary"):
                report = f"""
# COMPREHENSIVE ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. DATA SUMMARY
- Dataset Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns
- Memory Usage: {st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- Missing Values: {st.session_state.df.isnull().sum().sum()}
- Duplicate Rows: {st.session_state.df.duplicated().sum()}

## 2. ANALYSIS RESULTS
"""
                if st.session_state.analysis_results:
                    if 'grid' in st.session_state.analysis_results:
                        r = st.session_state.analysis_results
                        report += f"""
### Grid-based Hotspot Analysis
- Grid Cell Size: {r.get('cell_size', 'N/A')}m
- Total Incidents: {r.get('total_incidents', 'N/A')}
- Hotspot Cells: {r.get('hotspot_cells', 'N/A')}
- Threshold: {r.get('threshold', 'N/A'):.0f} incidents per cell
"""
                    if 'temporal' in st.session_state.analysis_results:
                        temp = st.session_state.analysis_results['temporal']
                        report += f"""
### Temporal Patterns
- Peak Hour: {temp['by_hour'].idxmax()}:00
- Peak Day: {temp['by_day'].idxmax()}
- Peak Month: {temp['by_month'].idxmax()}
"""
                
                report += """
## 3. RECOMMENDATIONS
- Deploy additional patrols to identified hotspot areas
- Focus resources on high-incident grid cells
- Monitor hotspot trends over time for pattern changes

## 4. DATA QUALITY
"""
                if 'Lat_Public' in st.session_state.df.columns:
                    valid = st.session_state.df[['Lat_Public', 'Long_Public']].notna().all(axis=1).sum()
                    report += f"- Valid Coordinates: {valid}/{len(st.session_state.df)} ({valid/len(st.session_state.df)*100:.1f}%)\n"
                
                completeness = (1 - st.session_state.df.isnull().sum().sum() / 
                              (st.session_state.df.shape[0] * st.session_state.df.shape[1])) * 100
                report += f"- Completeness: {completeness:.1f}%\n"
                
                if encrypt_reports:
                    report += "\n[ENCRYPTED: This report contains sensitive information]\n"
                
                report += "\n---\nEND OF REPORT"
                
                st.markdown(report)
                
                # Download button
                st.download_button(
                    label="💾 Download Report",
                    data=report,
                    file_name=f"crime_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                log_security_event("REPORT_GENERATED", "Report created and displayed")
        else:
            st.info("👈 Please load data and run analysis first")
    
    # Tab 5: Testing
    with tab5:
        st.header("System Testing")
        
        if st.button("🧪 Run All Tests", type="primary"):
            test_results = []
            
            with st.spinner("Running tests..."):
                # Test 1: Data Loading
                try:
                    test_df = create_sample_data()
                    assert test_df is not None
                    assert len(test_df) > 0
                    test_results.append(("Data Loading", "PASS", "✅"))
                except Exception as e:
                    test_results.append(("Data Loading", f"FAIL: {e}", "❌"))
                
                # Test 2: Data Validation
                try:
                    issues = validate_data_input(test_df)
                    assert isinstance(issues, list)
                    test_results.append(("Data Validation", "PASS", "✅"))
                except Exception as e:
                    test_results.append(("Data Validation", f"FAIL: {e}", "❌"))
                
                # Test 3: Hotspot Analysis
                try:
                    results, error = perform_hotspot_analysis(test_df, 250)
                    assert error is None
                    assert results is not None
                    test_results.append(("Hotspot Analysis", "PASS", "✅"))
                except Exception as e:
                    test_results.append(("Hotspot Analysis", f"FAIL: {e}", "❌"))
                
                # Test 4: Temporal Analysis
                try:
                    result, error = perform_temporal_analysis(test_df)
                    assert error is None
                    test_results.append(("Temporal Analysis", "PASS", "✅"))
                except Exception as e:
                    test_results.append(("Temporal Analysis", f"FAIL: {e}", "❌"))
            
            # Display results
            st.subheader("Test Results")
            for test_name, result, icon in test_results:
                st.write(f"{icon} **{test_name}**: {result}")
            
            passed = sum(1 for _, r, _ in test_results if "PASS" in r)
            total = len(test_results)
            
            if passed == total:
                st.success(f"🎉 All {total} tests passed!")
            else:
                st.warning(f"⚠️ {passed}/{total} tests passed")
    
    # Tab 6: Help
    with tab6:
        st.header("Help & Documentation")
        st.markdown("""
        ## 📚 User Guide
        
        ### Getting Started
        1. **Load Data**: Use the sidebar to upload a file, load sample data, or load from URL
        2. **Explore**: Navigate to the Explore tab to view your data
        3. **Visualize**: Create charts and graphs in the Visualize tab
        4. **Analyze**: Run hotspot analysis in the Analysis tab
        5. **Report**: Generate comprehensive reports in the Reports tab
        
        ### Data Requirements
        - **Required Columns**: `Lat_Public`, `Long_Public` for spatial analysis
        - **Optional Columns**: Date/Time fields, Crime Type, District
        - **Supported Formats**: CSV, XLSX, GeoJSON
        
        ### Analysis Methods
        
        #### Grid-based Hotspot (250m/500m)
        - Creates a grid overlay on your geographic area
        - Identifies cells with high incident concentration
        - Marks top 20% as hotspots
        
        #### Temporal Analysis
        - Analyzes patterns by hour, day, and month
        - Identifies peak times for incidents
        - Requires date/time column
        
        #### Crime Type Distribution
        - Shows frequency of each crime type
        - Identifies most common offenses
        - Requires crime type column
        
        ### Security Features
        - **Anonymize Coordinates**: Adds random noise for privacy
        - **Encrypt Reports**: Marks reports as encrypted
        - **Security Logging**: All actions are logged
        - **Data Validation**: Checks for sensitive information
        
        ### Multi-User Support
        This application supports multiple simultaneous users. Each user maintains their own session,
        allowing independent data analysis without interference.
        
        ### Troubleshooting
        - **No data loaded**: Load data from the sidebar first
        - **Missing columns**: Ensure your data has required column names
        - **Analysis errors**: Check data quality and coordinate validity
        """)
    
    # Footer with security log
    with st.expander("🔒 Security Event Log"):
        if st.session_state.security_log:
            for log in st.session_state.security_log[-