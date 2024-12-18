import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, text
from sqlalchemy.exc import SQLAlchemyError

# 데이터베이스 연결 정보
db_user = 'root'
db_password = '1234'
db_host = 'localhost'
db_port = '3306'
sim_db_name = 'simulation_result'

sim_db_connection_str = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{sim_db_name}'
sim_db_connection = create_engine(sim_db_connection_str)
sim_table_name = 'episode_data'

label_db_name = 'query_result'

label_db_connection_str = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{label_db_name}'
label_db_connection = create_engine(label_db_connection_str)
label_table_name = 'labeled_data'

def save_labeled_data_to_db(df):
    try:
        df.to_sql(
            name=label_table_name,
            con=label_db_connection,
            if_exists='append',
            index=False
        )
    except Exception as e:
        print(f"데이터 저장 중 오류가 발생했습니다: {e}")
def load_whole_labeled_data_from_db():
    """
    지정된 테이블의 모든 데이터를 DataFrame으로 가져오는 함수.

    Parameters:
    - engine: SQLAlchemy 엔진 객체
    - table_name (str): 데이터를 가져올 테이블의 이름

    Returns:
    - pandas.DataFrame: 테이블의 모든 데이터를 포함한 DataFrame
    """
    query = f"SELECT * FROM {label_table_name};"
    try:
        df = pd.read_sql_query(query, label_db_connection)
        print(f"테이블 '{label_table_name}'에서 모든 데이터를 성공적으로 불러왔습니다.")
        return df
    except SQLAlchemyError as e:
        print(f"데이터 불러오기 중 오류 발생: {e}")


def get_labeled_data_record_count():
    """
    지정된 테이블의 레코드 수를 반환하는 함수.

    Returns:
    - int: 테이블의 총 레코드 수
    """
    query = f"SELECT COUNT(*) AS row_count FROM {label_table_name};"
    try:
        with label_db_connection.connect() as connection:
            result = connection.execute(text(query))
            count = result.scalar()  # COUNT(*) 결과를 직접 가져옴
            print(f"테이블 '{label_table_name}'의 레코드 수: {count}")
            return count
    except SQLAlchemyError as e:
        print(f"레코드 수 조회 중 오류 발생: {e}")
        return None

def delete_labeled_data_from_db():
    """
    지정된 테이블의 모든 데이터를 삭제하는 함수.

    Returns:
    - None
    """
    metadata = MetaData()
    try:
        table = Table(label_table_name, metadata, autoload_with=label_db_connection)
        table.drop(label_db_connection, checkfirst=True)
        print(f"테이블 '{label_table_name}'이(가) 성공적으로 삭제되었습니다.")
    except SQLAlchemyError as e:
        print(f"테이블 '{label_table_name}' 삭제 중 오류 발생: {e}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")


def save_episode_data_to_db(df, episode_id):
    # 'episode_id' 컬럼 추가
    df['episode_id'] = episode_id  # 에피소드 ID 설정
    try:
        df.to_sql(
            name=sim_table_name,
            con=sim_db_connection,
            if_exists='append',
            index=False
        )
    except Exception as e:
        print(f"데이터 저장 중 오류가 발생했습니다: {e}")


def load_episode_data_from_db(episode_id):
    query = f"""
        SELECT *
        FROM episode_data
        WHERE episode_id = {episode_id};
    """
    try:
        df = pd.read_sql_query(query, sim_db_connection)
        df.drop(columns=['episode_id'], inplace=True)
        return df
    except Exception as e:
        print("데이터 조회 중 오류 발생:", e)
def delete_episode_data_from_db():
    """
    지정된 테이블을 삭제(drop)하는 함수.

    Parameters:
    - engine: SQLAlchemy 엔진 객체
    - table_name (str): 삭제할 테이블의 이름

    Returns:
    - None
    """
    metadata = MetaData()
    try:
        table = Table(sim_table_name, metadata, autoload_with=sim_db_connection)
        table.drop(sim_db_connection, checkfirst=True)
        print(f"테이블 '{sim_table_name}'이(가) 성공적으로 삭제되었습니다.")
    except SQLAlchemyError as e:
        print(f"테이블 '{sim_table_name}' 삭제 중 오류 발생: {e}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")

def get_simulation_result_record_count():
    """
    지정된 테이블의 레코드 수를 반환하는 함수.

    Returns:
    - int: 테이블의 총 레코드 수
    """
    query = f"SELECT COUNT(*) AS row_count FROM {sim_table_name};"
    try:
        with sim_db_connection.connect() as connection:
            result = connection.execute(text(query))
            count = result.scalar()
        print(f"테이블 '{sim_table_name}'의 레코드 수: {count}")
        return count
    except SQLAlchemyError as e:
        print(f"레코드 수 조회 중 오류 발생: {e}")
        return None
