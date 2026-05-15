import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import traceback

# 🌟 지도 시각화를 위한 geopandas 추가
import geopandas as gpd

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def find_newest_file(filename, search_dir):
    """이름이 같은 파일이 여러 개면, 가장 최신 파일을 찾습니다."""
    found_files = []
    for root, dirs, files in os.walk(search_dir):
        if filename in files:
            found_files.append(os.path.join(root, filename))
    if not found_files: return None
    return max(found_files, key=os.path.getmtime)

def find_shapefile(search_dir):
    """서울시 행정동 경계 지도 파일(.shp)을 찾습니다."""
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            # 팀원 파일명 규칙(bnd_dong_11_어쩌구.shp) 호환
            if file.endswith(".shp") and "dong" in file.lower() and "11" in file:
                return os.path.join(root, file)
    return None

def main():
    print("🚀 1, 2번 파일 연계 및 지도 시각화 프로그램을 시작합니다...")
    
    try:
        search_dir = r"C:\Users\user\Desktop\찐찐찐막\어딜가든"
        if not os.path.exists(search_dir):
            search_dir = os.getcwd()
            
        gu_file = find_newest_file("01_gu_scored.csv", search_dir)
        dong_file = find_newest_file("02_dong_scored.csv", search_dir)
        shp_file = find_shapefile(search_dir)
        
        if not gu_file:
            print("❌ '01_gu_scored.csv' 파일을 찾지 못했습니다.")
            return
            
        print(f"✅ 최신 구 단위 데이터 발견!: {gu_file}")
        df_gu = pd.read_csv(gu_file, encoding='utf-8-sig')
        
        available_cols = {
            'crime_data': '범죄건수', 'crime_rate': '범죄발생률',
            'pop_tot': '생활인구', 'single_cnt': '1인가구수', 
            'entertainment_cnt': '유흥시설수', 'incon_report': '불편신고건수',
            'infra_cnt': '보안인프라(CCTV)', 'safe_route_acc': '안심귀갓길'
        }
        
        plot_cols = [c for c in available_cols.keys() if c in df_gu.columns]
        plot_labels = [available_cols[c] for c in plot_cols]
        risk_col = 'risk_sc' if 'risk_sc' in df_gu.columns else 'risk_score'

        for c in plot_cols + [risk_col]:
            if c in df_gu.columns:
                df_gu[c] = pd.to_numeric(df_gu[c].astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce').fillna(0)

        output_dir = os.path.join(os.getcwd(), "output_images")
        os.makedirs(output_dir, exist_ok=True)

        print("📊 이미지 저장 중... (화면에는 표시되지 않고 바로 폴더에 저장됩니다)")
        
        # --- 시각화 1: 자치구별 위험 점수 막대 그래프 ---
        if risk_col in df_gu.columns:
            plt.figure(figsize=(12, 8))
            df_gu_sorted = df_gu.sort_values(risk_col, ascending=False)
            sns.barplot(x=risk_col, y='GU_NM', data=df_gu_sorted, palette='Reds_r')
            plt.title('서울시 자치구별 야간 보행 위험 점수', fontsize=16, fontweight='bold')
            plt.xlabel('잠재위험지수 (risk_sc)', fontsize=12)
            plt.ylabel('자치구', fontsize=12)
            plt.savefig(os.path.join(output_dir, "01_GU_Risk_Score_BarChart.png"), dpi=300, bbox_inches='tight')
            plt.close() 

        # --- 시각화 2: 지표 간 상관관계 히트맵 ---
        if len(plot_cols) > 1 and risk_col in df_gu.columns:
            plt.figure(figsize=(12, 10))
            corr_data = df_gu[plot_cols + [risk_col]].copy()
            corr_data.columns = plot_labels + ['잠재위험지수']
            sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('구 단위 위험 지표 간 상관관계 분석', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.savefig(os.path.join(output_dir, "02_GU_Correlation_Heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()

        # --- 시각화 3: 상위 자치구 방사형 차트 (거미줄) ---
        if len(plot_cols) >= 3 and risk_col in df_gu.columns:
            plt.figure(figsize=(12, 10))
            top_3_df = df_gu.sort_values(risk_col, ascending=False).head(3)
            radar_cols = plot_cols
            radar_labels = plot_labels
            norm_data = top_3_df[radar_cols].copy()
            for col in radar_cols:
                min_val, max_val = df_gu[col].min(), df_gu[col].max()
                norm_data[col] = (norm_data[col] - min_val) / (max_val - min_val) if max_val != min_val else 0
            
            angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
            angles += angles[:1]
            radar_labels += radar_labels[:1]

            ax = plt.subplot(111, polar=True)
            colors = ['#FF4C4C', '#FF9F43', '#1AB1C9']
            for i, (index, row) in enumerate(norm_data.iterrows()):
                gu_nm = top_3_df.loc[index, 'GU_NM']
                data = row.tolist() + row.tolist()[:1]
                ax.plot(angles, data, color=colors[i], linewidth=2, linestyle='solid', label=gu_nm)
                ax.fill(angles, data, color=colors[i], alpha=0.2)
                
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            plt.xticks(angles[:-1], radar_labels[:-1], fontsize=11, fontweight='bold')
            ax.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75, 1.0], ['0.25', '0.50', '0.75', '1.00'], color='grey', fontsize=10)
            plt.title('서울시 위험도 상위 자치구별 지표 비교 (상세)', fontsize=18, fontweight='bold', y=1.08)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
            plt.savefig(os.path.join(output_dir, "03_GU_Risk_Factors_RadarChart.png"), dpi=300, bbox_inches='tight')
            plt.close()

        # --- 시각화 4 & 5: 동 단위 막대 그래프 및 🌟지도 시각화🌟 ---
        if dong_file:
            print(f"✅ 최신 동 단위 데이터 발견!: {dong_file}")
            df_dong = pd.read_csv(dong_file, encoding='utf-8-sig')
            
            if risk_col in df_dong.columns:
                df_dong[risk_col] = pd.to_numeric(df_dong[risk_col].astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors='coerce').fillna(0)
            
            # (4) 동 단위 상위 15개 막대 그래프
            if risk_col in df_dong.columns and 'GU_NM' in df_dong.columns and 'DONG_NM' in df_dong.columns:
                plt.figure(figsize=(12, 8))
                df_dong['지역명'] = df_dong['GU_NM'] + ' ' + df_dong['DONG_NM']
                df_dong_sorted = df_dong.sort_values(risk_col, ascending=False).head(15)
                
                sns.barplot(x=risk_col, y='지역명', data=df_dong_sorted, palette='Oranges_r')
                plt.title('서울시 잠재 위험 상위 15개 행정동', fontsize=16, fontweight='bold')
                plt.xlabel('잠재위험지수 (risk_sc)', fontsize=12)
                plt.ylabel('행정동', fontsize=12)
                for i, v in enumerate(df_dong_sorted[risk_col]):
                    plt.text(v + 0.005, i, f"{v:.3f}", va='center', fontsize=10)
                plt.savefig(os.path.join(output_dir, "04_DONG_Risk_Score_BarChart.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
            # 🌟 (5) 동 단위 위험도 컬러 지도 (Choropleth Map) 🌟
            if shp_file and 'DONG_CD' in df_dong.columns:
                print(f"✅ 지도 경계 데이터 발견!: {shp_file}")
                
                # 지도 데이터 로딩 및 인코딩
                dong_gdf = gpd.read_file(shp_file, encoding='utf-8') # 혹은 utf-8
                
                # 컬럼 통일화 (SHP 파일의 ADM_CD를 DONG_CD로 변경)
                cd_col = next((c for c in dong_gdf.columns if 'CD' in c.upper()), None)
                if cd_col:
                    dong_gdf['DONG_CD'] = dong_gdf[cd_col].astype(str).str[:8]
                    df_dong['DONG_CD'] = df_dong['DONG_CD'].astype(str).str[:8]
                    
                    # 지도 데이터와 위험도 데이터 합치기
                    merged_gdf = dong_gdf.merge(df_dong, on='DONG_CD', how='inner')
                    
                    # 지도 그리기
                    plt.figure(figsize=(15, 12))
                    ax = plt.gca()
                    
                    # 빨간색 계열(Reds)로 칠하며, 점수가 높을수록 진하게 표시
                    merged_gdf.plot(
                        column=risk_col, cmap='Reds', linewidth=0.5, 
                        ax=ax, edgecolor='0.5', legend=True,
                        legend_kwds={'label': "잠재위험지수 (risk_sc)", 'orientation': "vertical", 'shrink': 0.7}
                    )
                    
                    plt.title('서울시 행정동별 야간 보행 잠재 위험도 지도', fontsize=20, fontweight='bold')
                    ax.set_axis_off() # 거추장스러운 위경도 눈금선 제거
                    
                    plt.savefig(os.path.join(output_dir, "05_Seoul_Dong_Risk_Map.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                    print("✅ 지도 시각화 이미지 생성 완료: 05_Seoul_Dong_Risk_Map.png")
            else:
                print("⚠️ SHP 지도 파일을 찾지 못했거나 DONG_CD가 없어 지도를 그릴 수 없습니다.")
        else:
            print("⚠️ '02_dong_scored.csv' 파일을 찾지 못해 동 단위 시각화는 생략합니다.")

        print("\n🎉 모든 시각화 작업이 성공적으로 끝났습니다! output_images 폴더를 확인해주세요.")

    except Exception as e:
        print("\n🚨 프로그램 실행 중 에러가 발생했습니다! 아래 내용을 확인해주세요:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()