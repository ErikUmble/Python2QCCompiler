"""
Quantum Benchmarking Database Library (Simplified Single-Problem Design)

A unified database interface for quantum circuit benchmarking, designed for
single problem types per database. Each problem type (3SAT, Clique, etc.)
should have its own directory with a dedicated database.

Database Schema (per problem type):
- problem_instances: Stores unique problem instances of one type
- trials: Stores trial results referencing problem instances by ID

Key Features:
- Single problem type per database (simplified design)
- Normalized database (no duplicate problem storage across trials)
- Abstract oracle method for centralized circuit generation
- Async job result fetching from IBM Quantum
- Comprehensive documentation and maintenance tools

"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

from .types import BaseTrial, ProblemInstance

# Configure logging
logger = logging.getLogger("benchmarklib.core.database")


class BenchmarkDatabase:
    """
    Database manager for a single quantum problem type.

    This class manages both problem instances and trials for ONE problem type
    in a normalized database schema. Each problem type should have its own
    database instance and file.

    Database Tables:
    - problem_instances: Stores unique problem instances of this type
    - trials: Stores trial results referencing problem instances

    Type Safety:
    The database expects to work with consistent ProblemInstance and BaseTrial
    subclasses. Register these types when creating the database.
    """

    def __init__(
        self,
        db_name: str,
        problem_class: Type[ProblemInstance],
        trial_class: Type[BaseTrial],
    ):
        """
        Initialize database for a specific problem type.

        Args:
            db_name: SQLite database filename
            problem_class: ProblemInstance subclass for this database
            trial_class: BaseTrial subclass for this database
        """
        self.db_name = db_name
        self.problem_class = problem_class
        self.trial_class = trial_class

        # Get problem type from the class
        dummy_problem = problem_class.__new__(problem_class)
        dummy_problem.instance_id = None
        self.problem_type = dummy_problem.problem_type

        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        """Create database connection with proper configuration."""
        conn = sqlite3.connect(self.db_name, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety/speed
        conn.execute("PRAGMA foreign_keys=ON")  # Enable foreign key constraints
        return conn

    def _initialize_database(self) -> None:
        """Create database tables if they don't exist."""
        with self._connect() as conn:
            cursor = conn.cursor()

            # Create problem_instances table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS problem_instances (
                    instance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    problem_type TEXT NOT NULL DEFAULT '{self.problem_type}',
                    problem_data TEXT NOT NULL,
                    size_metrics TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Create trials table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance_id INTEGER NOT NULL,
                    compiler_name TEXT NOT NULL,
                    job_id TEXT,
                    job_pub_idx INTEGER NOT NULL DEFAULT 0,
                    counts TEXT,
                    simulation_counts TEXT,
                    trial_params TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (instance_id) REFERENCES problem_instances (instance_id)
                        ON DELETE CASCADE
                )
            """)

            # Create indices for performance
            indices = [
                "CREATE INDEX IF NOT EXISTS idx_problem_size ON problem_instances (size_metrics)",
                "CREATE INDEX IF NOT EXISTS idx_trials_instance ON trials (instance_id)",
                "CREATE INDEX IF NOT EXISTS idx_trials_job ON trials (job_id)",
                "CREATE INDEX IF NOT EXISTS idx_trials_compile ON trials (compiler_name)",
                "CREATE INDEX IF NOT EXISTS idx_trials_pending ON trials (job_id, counts)",
            ]

            for index_sql in indices:
                cursor.execute(index_sql)

            conn.commit()
            logger.info(f"Database initialized: {self.db_name} ({self.problem_type})")

    # Problem Instance Operations
    def save_problem_instance(self, problem: ProblemInstance) -> int:
        """
        Save problem instance to database.

        Args:
            problem: Problem instance to save (must match registered type)

        Returns:
            Instance ID (sets problem.instance_id as side effect)

        Raises:
            TypeError: If problem is not of the expected type
        """
        if not isinstance(problem, self.problem_class):
            raise TypeError(
                f"Expected {self.problem_class.__name__}, got {type(problem).__name__}"
            )

        now = datetime.now().isoformat()
        problem_data = json.dumps(problem.to_dict())
        size_metrics = json.dumps(problem.get_problem_size())

        with self._connect() as conn:
            cursor = conn.cursor()

            if problem.instance_id is None:
                # Insert new problem
                cursor.execute(
                    """
                    INSERT INTO problem_instances 
                    (problem_type, problem_data, size_metrics, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (self.problem_type, problem_data, size_metrics, now, now),
                )

                problem.instance_id = cursor.lastrowid
                logger.debug(f"Saved new problem instance: {problem}")
            else:
                # Update existing problem
                cursor.execute(
                    """
                    UPDATE problem_instances 
                    SET problem_data=?, size_metrics=?, updated_at=?
                    WHERE instance_id=?
                """,
                    (problem_data, size_metrics, now, problem.instance_id),
                )

                logger.debug(f"Updated problem instance: {problem}")

            conn.commit()

        return problem.instance_id

    def get_problem_instance(self, instance_id: int) -> ProblemInstance:
        """
        Retrieve problem instance by ID.

        Args:
            instance_id: Database ID of problem instance

        Returns:
            Problem instance object of the registered type

        Raises:
            ValueError: If instance not found
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT problem_data FROM problem_instances 
                WHERE instance_id = ?
            """,
                (instance_id,),
            )

            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Problem instance {instance_id} not found")

            problem_data_json = row[0]
            problem_data = json.loads(problem_data_json)

            return self.problem_class.from_dict(
                data=problem_data, instance_id=instance_id
            )

    def find_problem_instances(
        self,
        size_filters: Optional[Dict[str, Any]] = None,
        problem_data_filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        choose_untested: bool = False,
        random_sample: bool = False,
        compiler: Optional["SynthesisCompiler"] = None,
    ) -> List[ProblemInstance]:
        """
        Find problem instances matching criteria.

        Args:
            size_filters: Filter by size metrics (e.g., {"num_vars": 10})
            limit: Maximum number of results
            choose_untested: If True, only return problems with no trials (for specified compiler if provided)
            random_sample: If True, randomly sample from filtered results
            compiler: When choose_untested=True, find instances untested by this specific compiler

        Returns:
            List of matching problem instances
        """

        # Build base query
        if choose_untested:
            if compiler:
                # Find problems with no trials for this specific compiler
                query = """
                SELECT DISTINCT p.instance_id, p.problem_data, p.size_metrics
                FROM problem_instances p
                LEFT JOIN trials t ON p.instance_id = t.instance_id 
                    AND t.compiler_name = ?
                WHERE t.instance_id IS NULL
                """
                params = [compiler.name]
            else:
                # Find problems with no trials at all
                query = """
                SELECT p.instance_id, p.problem_data, p.size_metrics
                FROM problem_instances p
                LEFT JOIN trials t ON p.instance_id = t.instance_id
                WHERE t.instance_id IS NULL
                """
                params = []
        else:
            query = (
                "SELECT instance_id, problem_data, size_metrics FROM problem_instances"
            )
            params = []

        # Apply size filters
        if size_filters or problem_data_filters:
            size_filters = size_filters or {}
            problem_data_filters = problem_data_filters or {}
            
            where_conditions = []
            for key, value in size_filters.items():
                if choose_untested:
                    where_conditions.append(
                        f"JSON_EXTRACT(p.size_metrics, '$.{key}') = ?"
                    )
                else:
                    where_conditions.append(
                        f"JSON_EXTRACT(size_metrics, '$.{key}') = ?"
                    )
                params.append(value)

            for key, value in problem_data_filters.items():
                if choose_untested:
                    where_conditions.append(
                        f"JSON_EXTRACT(p.problem_data, '$.{key}') = ?"
                    )
                else:
                    where_conditions.append(
                        f"JSON_EXTRACT(problem_data, '$.{key}') = ?"
                    )
                params.append(value)

            if where_conditions:
                if choose_untested:
                    query += " AND " + " AND ".join(where_conditions)
                else:
                    query += " WHERE " + " AND ".join(where_conditions)

        # Add random ordering if requested
        if random_sample:
            query += " ORDER BY RANDOM()"

        # Apply limit
        if limit:
            query += f" LIMIT {limit}"

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        # Reconstruct problem instances
        instances = []
        for instance_id, prob_data_json, size_metrics_json in rows:
            problem_data = json.loads(prob_data_json)
            instance = self.problem_class.from_dict(problem_data, instance_id)
            instances.append(instance)

        return instances

    # Trial Operations
    def save_trial(self, trial: BaseTrial) -> int:
        """
        Save trial to database.

        Args:
            trial: Trial to save (must match registered type)

        Returns:
            Trial ID (sets trial.trial_id as side effect)

        Raises:
            TypeError: If trial is not of the expected type
        """
        if not isinstance(trial, self.trial_class):
            raise TypeError(
                f"Expected {self.trial_class.__name__}, got {type(trial).__name__}"
            )

        now = datetime.now().isoformat()
        counts_json = json.dumps(trial.counts) if trial.counts else ""
        sim_counts_json = (
            json.dumps(trial.simulation_counts) if trial.simulation_counts else ""
        )
        trial_params_json = json.dumps(trial.trial_params)

        with self._connect() as conn:
            cursor = conn.cursor()

            if trial.trial_id is None:
                # Insert new trial
                cursor.execute(
                    """
                    INSERT INTO trials 
                    (instance_id, compiler_name, job_id, job_pub_idx, counts, 
                     simulation_counts, trial_params, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        trial.instance_id,
                        trial.compiler_name,
                        trial.job_id,
                        trial.job_pub_idx,
                        counts_json,
                        sim_counts_json,
                        trial_params_json,
                        trial.created_at,
                        now,
                    ),
                )

                trial.trial_id = cursor.lastrowid
                logger.debug(f"Saved new trial ID: {trial.trial_id}")
            else:
                # Update existing trial
                cursor.execute(
                    """
                    UPDATE trials 
                    SET instance_id=?, compiler_name=?, job_id=?, job_pub_idx=?, 
                        counts=?, simulation_counts=?, trial_params=?, updated_at=?
                    WHERE trial_id=?
                """,
                    (
                        trial.instance_id,
                        trial.compiler_name,
                        trial.job_id,
                        trial.job_pub_idx,
                        counts_json,
                        sim_counts_json,
                        trial_params_json,
                        now,
                        trial.trial_id,
                    ),
                )

                logger.debug(f"Updated trial ID: {trial.trial_id}")

            conn.commit()

        return trial.trial_id

    def get_trial(self, trial_id: int) -> BaseTrial:
        """
        Retrieve trial by ID.

        Args:
            trial_id: Database ID of trial

        Returns:
            Trial object of the registered type

        Raises:
            ValueError: If trial not found
        """
        trials = self.find_trials(trial_id=trial_id)
        if not trials:
            raise ValueError(f"Trial {trial_id} not found")
        return trials[0]

    def find_trials(
        self,
        trial_id: Optional[int] = None,
        instance_id: Optional[int] = None,
        job_id: Optional[str] = None,
        compiler_name: Optional[str] = None,
        include_pending: bool = True,
        trial_params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[BaseTrial]:
        """
        Find trials matching criteria.

        Args:
            trial_id: Specific trial ID
            instance_id: Filter by problem instance
            job_id: Filter by IBM Quantum job ID
            compiler_name: Filter by compilation method
            include_pending: Include trials without results
            trial_params: Filter by trial parameters
            limit: Maximum number of results

        Returns:
            List of matching trials
        """
        query_parts = []
        params = []

        base_query = """
            SELECT t.trial_id, t.instance_id, t.compiler_name, t.job_id, t.job_pub_idx,
                   t.counts, t.simulation_counts, t.trial_params, t.created_at,
                   p.problem_data
            FROM trials t
            JOIN problem_instances p ON t.instance_id = p.instance_id
        """

        if trial_id is not None:
            query_parts.append("t.trial_id = ?")
            params.append(trial_id)

        if instance_id is not None:
            query_parts.append("t.instance_id = ?")
            params.append(instance_id)

        if job_id is not None:
            query_parts.append("t.job_id = ?")
            params.append(job_id)

        if compiler_name is not None:
            query_parts.append("t.compiler_name = ?")
            params.append(compiler_name)

        if not include_pending:
            query_parts.append("t.counts != ''")

        # Build final query
        query = base_query
        if query_parts:
            query += " WHERE " + " AND ".join(query_parts)

        if limit:
            query += f" LIMIT {limit}"

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()


        # Reconstruct trial objects
        trials = []
        for row in rows:
            (
                trial_id,
                instance_id,
                compiler_name,
                job_id,
                job_pub_idx,
                counts_json,
                sim_counts_json,
                trial_params_json,
                created_at,
                problem_data_json,
            ) = row

            # Deserialize JSON fields
            counts = json.loads(counts_json) if counts_json else None
            sim_counts = json.loads(sim_counts_json) if sim_counts_json else None
            params_dict = json.loads(trial_params_json)


            problem_data = json.loads(problem_data_json)
            problem_instance = self.problem_class.from_dict(
                data=problem_data, instance_id=instance_id
            )

            # Create trial object
            trial = self.trial_class(
                problem_instance=problem_instance,
                compiler_name=compiler_name,
                job_id=job_id,
                job_pub_idx=job_pub_idx,
                counts=counts,
                simulation_counts=sim_counts,
                trial_id=trial_id,
                created_at=created_at,
                **params_dict,
            )

            # Apply trial parameter filters if specified
            if trial_params:
                if all(trial.trial_params.get(k) == v for k, v in trial_params.items()):
                    trials.append(trial)
            else:
                trials.append(trial)

        return trials

    def delete_trial(self, trial_id: int) -> None:
        """Delete trial from database."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM trials WHERE trial_id = ?", (trial_id,))
            conn.commit()
        logger.info(f"Deleted trial ID: {trial_id}")

    def delete_problem_instance(self, instance_id: int) -> None:
        """
        Delete problem instance and all associated trials.

        Args:
            instance_id: ID of problem instance to delete
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            # Foreign key constraints will cascade delete trials
            cursor.execute(
                "DELETE FROM problem_instances WHERE instance_id = ?", (instance_id,)
            )
            conn.commit()
        logger.info(f"Deleted problem instance ID: {instance_id}")

    # Async Job Management
    def get_pending_job_ids(self) -> List[str]:
        """Get all job IDs with pending results."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT job_id FROM trials 
                WHERE job_id IS NOT NULL AND counts = ''
            """)
            return [row[0] for row in cursor.fetchall()]

    async def update_job_results(self, job_id: str, service) -> None:
        """
        Fetch and update results for a specific job.

        Args:
            job_id: IBM Quantum job ID
            service: QiskitRuntimeService instance
        """
        try:
            # Fetch job results asynchronously
            retrieved_job = await asyncio.to_thread(service.job, job_id)
            results = await asyncio.to_thread(retrieved_job.result)

            # Update all trials for this job
            trials = self.find_trials(job_id=job_id, include_pending=True)
            updated_count = 0

            for trial in trials:
                if trial.is_pending:
                    # Extract counts from results
                    pub_result = results[trial.job_pub_idx]

                    # Handle different result data structures
                    if hasattr(pub_result.data, "c"):
                        counts = pub_result.data.c.get_counts()
                    elif hasattr(pub_result.data, "meas"):
                        counts = pub_result.data.meas.get_counts()
                    else:
                        # Fallback - mark as failed
                        trial.mark_failure()
                        counts = trial.counts

                    trial.counts = counts
                    self.save_trial(trial)
                    updated_count += 1

            logger.info(f"Updated {updated_count} trials for job {job_id}")

        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
            # Mark trials as failed
            trials = self.find_trials(job_id=job_id, include_pending=True)
            for trial in trials:
                if trial.is_pending:
                    trial.mark_failure()
                    self.save_trial(trial)

    async def update_all_pending_results(self, service, batch_size: int = 5) -> None:
        """
        Update all pending job results asynchronously.

        Args:
            service: QiskitRuntimeService instance
            batch_size: Number of concurrent job fetches
        """
        pending_jobs = self.get_pending_job_ids()

        if not pending_jobs:
            logger.info("No pending jobs to update")
            return

        logger.info(f"Updating {len(pending_jobs)} pending jobs")

        # Process jobs in batches to avoid overwhelming the API
        for i in range(0, len(pending_jobs), batch_size):
            batch = pending_jobs[i : i + batch_size]
            tasks = [self.update_job_results(job_id, service) for job_id in batch]

            batch_num = i // batch_size + 1
            total_batches = (len(pending_jobs) + batch_size - 1) // batch_size
            logger.info(f"Processing batch {batch_num}/{total_batches}")

            await asyncio.gather(*tasks, return_exceptions=True)

    # Statistics and Maintenance
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        with self._connect() as conn:
            cursor = conn.cursor()

            # Problem instance stats
            cursor.execute("SELECT COUNT(*) FROM problem_instances")
            total_problems = cursor.fetchone()[0]

            # Trial stats
            cursor.execute("SELECT COUNT(*) FROM trials")
            total_trials = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM trials WHERE counts = ''")
            pending_trials = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM trials WHERE counts LIKE '%\"-1\"%'")
            failed_trials = cursor.fetchone()[0]

            cursor.execute("""
                SELECT compiler_name, COUNT(*) 
                FROM trials 
                GROUP BY compiler_name
            """)
            compile_stats = dict(cursor.fetchall())

            # Average trials per problem
            avg_trials_per_problem = (
                total_trials / total_problems if total_problems > 0 else 0
            )

        return {
            "problem_type": self.problem_type,
            "problem_instances": total_problems,
            "trials": {
                "total": total_trials,
                "pending": pending_trials,
                "failed": failed_trials,
                "completed": total_trials - pending_trials - failed_trials,
                "by_compiler_name": compile_stats,
                "avg_per_problem": round(avg_trials_per_problem, 2),
            },
        }

    def calculate_trial_success_rate(self, trial: BaseTrial) -> float:
        """
        Calculate success rate for a trial, automatically loading problem instance.

        Args:
            trial: Trial to calculate success rate for

        Returns:
            Success rate between 0 and 1
        """
        if trial._problem_instance is None:
            trial._problem_instance = self.get_problem_instance(trial.instance_id)
        return trial.calculate_success_rate()

    def calculate_trial_expected_success_rate(self, trial: BaseTrial) -> float:
        """
        Calculate expected success rate for a trial, automatically loading problem instance.

        Args:
            trial: Trial to calculate expected success rate for

        Returns:
            Expected success rate between 0 and 1
        """
        if trial._problem_instance is None:
            trial._problem_instance = self.get_problem_instance(trial.instance_id)
        return trial.calculate_expected_success_rate()

    def get_trial_with_success_rates(
        self, trial_id: int
    ) -> Tuple[BaseTrial, float, float]:
        """
        Get trial with computed success rates.

        Args:
            trial_id: ID of trial to retrieve

        Returns:
            Tuple of (trial, actual_success_rate, expected_success_rate)
        """
        trial = self.get_trial(trial_id)
        actual_rate = self.calculate_trial_success_rate(trial)
        expected_rate = self.calculate_trial_expected_success_rate(trial)
        return trial, actual_rate, expected_rate

    def recompute_simulations(self, instance_ids: Optional[List[int]] = None) -> None:
        """
        Recompute simulation results for trials.

        Args:
            instance_ids: Limit to specific problem instances (None for all)
        """
        # Find trials to recompute
        if instance_ids:
            trials = []
            for instance_id in instance_ids:
                trials.extend(self.find_trials(instance_id=instance_id))
        else:
            trials = self.find_trials()

        logger.info(f"Recomputing simulations for {len(trials)} trials")

        for trial in trials:
            # Load problem instance and recompute simulation
            problem = self.get_problem_instance(trial.instance_id)

            # This would need to be implemented by specific trial types
            if hasattr(trial, "recompute_simulation"):
                trial.recompute_simulation(problem)
                self.save_trial(trial)


def hamming_distance(s1: str, s2: str) -> int:
    """Calculate Hamming distance between two binary strings."""
    if len(s1) != len(s2):
        raise ValueError("Strings must have equal length")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))
