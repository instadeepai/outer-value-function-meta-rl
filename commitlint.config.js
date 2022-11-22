module.exports = {
    extends: ['@commitlint/config-conventional'],
    rules: {
		'subject-case': [1, 'always'],
		'header-max-length': [1, 'always', 100],
		'type-enum': [
			2,
			'always',
			[
				'build',
				'chore',
				'ci',
				'docs',
				'feat',
				'fix',
				'perf',
				'refactor',
				'revert',
				'style',
				'test',
				'exp',
				'func',
			],
		],
	},
}
