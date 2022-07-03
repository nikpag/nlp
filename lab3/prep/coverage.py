def histogram(train_set):
	"""
	Create a histogram of sentence lengths for the training set

	Args:
		train_set: The training set data

	Returns: The desired histogram
	"""
	histogram = dict()
	for item in train_set:
		l = len(item)
		histogram[l] = histogram[l] + 1 if l in histogram else 1

	histogram = list(histogram.items())
	histogram = sorted(histogram)
	return histogram

def coverage(train_set, max_num):
	"""
	Calculate the training set's coverage (i.e. Percentage of
	words that don't have to be truncated), based on the maximum
	length given.

	Args:
		train_set: The training set data (or any set for that matter)
		max_num: The maximum length

	Returns: The coverage, formatted as a percentage
	"""
	tuple_list = histogram(train_set)
	total_sentences = sum([occur for (_, occur) in tuple_list])
	covered_sentences = sum([occur if max_num >= size else 0 for (size, occur) in tuple_list])
	return f"COVERAGE: {100 * covered_sentences / total_sentences}%"
