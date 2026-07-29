import sys
import re

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame

from pyspark.context import SparkContext
from pyspark.sql import functions as F


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BRONZE_PATH = (
    "s3://healthcare-project-dea-bronze-al/"
    "Daily-Nurse-Staffing/"
)

SILVER_PATH = (
    "s3://healthcare-project-dea-silver-al/"
    "Daily-Nurse-Staffing/"
)

ERROR_ROOT_PATH = (
    "s3://healthcare-project-dea-silver-transformation-errors-al/"
    "_errors/Daily-Nurse-Staffing/"
)


# ---------------------------------------------------------
# Initialize Glue
# ---------------------------------------------------------

args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME"],
)

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def normalize_column_name(column_name):
    """
    Convert column names to lowercase snake_case.
    """

    # Convert camel case to snake case
    normalized_name = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1_\2",
        column_name,
    )

    # Replace spaces and special characters
    normalized_name = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        normalized_name,
    )

    return normalized_name.strip("_").lower()


# ---------------------------------------------------------
# Read Bronze CSV
#
# The row-based reader is used so Glue can retain parser
# error records instead of terminating the job.
# ---------------------------------------------------------

bronze_dynamic_frame = (
    glueContext
    .create_dynamic_frame
    .from_options(
        connection_type="s3",
        connection_options={
            "paths": [BRONZE_PATH],
            "recurse": True,
        },
        format="csv",
        format_options={
            "withHeader": True,
            "separator": ",",
            "quoteChar": '"',
            "multiLine": True,
            "optimizePerformance": False,
        },
        transformation_ctx="nurse_staffing_bronze_source",
    )
)


# ---------------------------------------------------------
# Quarantine CSV parser errors
# ---------------------------------------------------------

parser_errors_dynamic_frame = (
    bronze_dynamic_frame
    .errorsAsDynamicFrame()
)

parser_error_count = parser_errors_dynamic_frame.count()

if parser_error_count > 0:
    glueContext.write_dynamic_frame.from_options(
        frame=parser_errors_dynamic_frame,
        connection_type="s3",
        connection_options={
            "path": (
                f"{ERROR_ROOT_PATH}"
                "parser_errors/"
            )
        },
        format="json",
        transformation_ctx="parser_error_target",
    )

    print(
        f"Quarantined {parser_error_count} "
        "CSV parser errors"
    )


# ---------------------------------------------------------
# Convert valid parsed rows to a Spark DataFrame
# ---------------------------------------------------------

staffing_df = bronze_dynamic_frame.toDF()


# ---------------------------------------------------------
# Normalize column names
# ---------------------------------------------------------

original_columns = staffing_df.columns

normalized_columns = [
    normalize_column_name(column_name)
    for column_name in original_columns
]

if len(normalized_columns) != len(set(normalized_columns)):
    raise ValueError(
        "Column-name normalization created duplicate names."
    )

for original_name, normalized_name in zip(
    original_columns,
    normalized_columns,
):
    staffing_df = staffing_df.withColumnRenamed(
        original_name,
        normalized_name,
    )


# Additional readable names
column_renames = {
    "provnum": "provider_number",
    "provname": "provider_name",
    "workdate": "work_date",
    "mdscensus": "mds_census",
}

for old_name, new_name in column_renames.items():
    if old_name in staffing_df.columns:
        staffing_df = staffing_df.withColumnRenamed(
            old_name,
            new_name,
        )


# ---------------------------------------------------------
# Validate required columns
# ---------------------------------------------------------

required_columns = {
    "cy_qtr",
    "work_date",
    "hrs_rndon",
}

missing_columns = (
    required_columns
    - set(staffing_df.columns)
)

if missing_columns:
    raise ValueError(
        "Required columns are missing: "
        f"{sorted(missing_columns)}"
    )


# ---------------------------------------------------------
# Add error-lineage fields
# ---------------------------------------------------------

staffing_df = staffing_df.withColumn(
    "_source_file",
    F.input_file_name(),
)

business_columns = [
    column_name
    for column_name in staffing_df.columns
    if column_name != "_source_file"
]

staffing_df = staffing_df.withColumn(
    "_raw_record",
    F.to_json(
        F.struct(
            *[
                F.col(column_name)
                for column_name in business_columns
            ]
        )
    ),
)


# ---------------------------------------------------------
# Prepare validation expressions
# ---------------------------------------------------------

quarter_raw = F.upper(
    F.trim(
        F.col("cy_qtr").cast("string")
    )
)

quarter_is_valid = quarter_raw.rlike(
    r"^\d{4}Q[1-4]$"
)

work_date_raw = F.trim(
    F.col("work_date").cast("string")
)

parsed_work_date = F.to_timestamp(
    work_date_raw,
    "yyyyMMdd",
)

hrs_rndon_position = (
    staffing_df.columns.index("hrs_rndon")
)

float_columns = [
    column_name
    for column_name in staffing_df.columns[
        hrs_rndon_position + 1:
    ]
    if not column_name.startswith("_")
]


# ---------------------------------------------------------
# Build validation error messages before casting
# ---------------------------------------------------------

error_expressions = [
    F.when(
        quarter_raw.isNull()
        | (quarter_raw == "")
        | (~quarter_is_valid),
        F.lit("Invalid cy_qtr; expected format YYYYQ1-YYYYQ4"),
    ),

    F.when(
        work_date_raw.isNull()
        | (work_date_raw == "")
        | parsed_work_date.isNull(),
        F.lit("Invalid work_date; expected format yyyyMMdd"),
    ),
]

for column_name in float_columns:
    numeric_raw = F.regexp_replace(
        F.trim(
            F.col(column_name).cast("string")
        ),
        ",",
        "",
    )

    parsed_numeric = numeric_raw.cast("double")

    error_expressions.append(
        F.when(
            numeric_raw.isNotNull()
            & (numeric_raw != "")
            & parsed_numeric.isNull(),
            F.lit(
                f"Invalid floating-point value: {column_name}"
            ),
        )
    )


staffing_df = staffing_df.withColumn(
    "_error_reason",
    F.concat_ws(
        "; ",
        *error_expressions,
    ),
)


# ---------------------------------------------------------
# Apply transformations
# ---------------------------------------------------------

# Convert 2024Q2 to 2024-Q2
staffing_df = staffing_df.withColumn(
    "cy_qtr",
    F.when(
        quarter_is_valid,
        F.regexp_replace(
            quarter_raw,
            r"^(\d{4})Q([1-4])$",
            "$1-Q$2",
        ),
    ),
)

# Convert 20240401 to 2024-04-01 00:00:00
staffing_df = staffing_df.withColumn(
    "work_date",
    parsed_work_date,
)

# Cast all columns after hrs_rndon to double
for column_name in float_columns:
    numeric_raw = F.regexp_replace(
        F.trim(
            F.col(column_name).cast("string")
        ),
        ",",
        "",
    )

    staffing_df = staffing_df.withColumn(
        column_name,
        F.when(
            numeric_raw.isNull()
            | (numeric_raw == ""),
            F.lit(None).cast("double"),
        ).otherwise(
            numeric_raw.cast("double")
        ),
    )


# ---------------------------------------------------------
# Split valid and invalid rows
# ---------------------------------------------------------

invalid_rows_df = staffing_df.filter(
    F.length(F.col("_error_reason")) > 0
)

valid_rows_df = (
    staffing_df
    .filter(
        F.length(F.col("_error_reason")) == 0
    )
    .drop(
        "_error_reason",
        "_raw_record",
        "_source_file",
    )
)


# ---------------------------------------------------------
# Write transformation errors to quarantine
# ---------------------------------------------------------

validation_error_count = invalid_rows_df.count()

if validation_error_count > 0:
    invalid_rows_dynamic_frame = DynamicFrame.fromDF(
        invalid_rows_df,
        glueContext,
        "nurse_staffing_validation_errors",
    )

    glueContext.write_dynamic_frame.from_options(
        frame=invalid_rows_dynamic_frame,
        connection_type="s3",
        connection_options={
            "path": (
                f"{ERROR_ROOT_PATH}"
                "validation_errors/"
            )
        },
        format="json",
        transformation_ctx="validation_error_target",
    )

    print(
        f"Quarantined {validation_error_count} "
        "validation errors"
    )


# ---------------------------------------------------------
# Write valid rows to Silver as Parquet
# ---------------------------------------------------------

valid_rows_dynamic_frame = DynamicFrame.fromDF(
    valid_rows_df,
    glueContext,
    "nurse_staffing_silver",
)

glueContext.write_dynamic_frame.from_options(
    frame=valid_rows_dynamic_frame,
    connection_type="s3",
    connection_options={
        "path": SILVER_PATH,
        "partitionKeys": [],
    },
    format="glueparquet",
    format_options={
        "compression": "snappy",
    },
    transformation_ctx="nurse_staffing_silver_target",
)


valid_rows_df.printSchema()

print(f"CSV parser errors: {parser_error_count}")
print(f"Validation errors: {validation_error_count}")


# Advance the bookmark only after every write succeeds
job.commit()
